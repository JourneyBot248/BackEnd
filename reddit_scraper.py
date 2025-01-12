import asyncpraw
import os
from dotenv import load_dotenv
import ollama
from pydantic import BaseModel, Field
import json

load_dotenv()

class ItineraryRequest(BaseModel):
    destination: str
    duration: int
    interests: list[str]

class LocationSummary(BaseModel):
    location: str = Field(..., description="The name of the location.")
    description: str = Field(..., description="A brief description of what the location offers.")

class RedditSummarizer:
    def __init__(self):
        self.reddit = None

    async def initialize(self):
        """Initializes the Reddit instance asynchronously."""
        self.reddit = await self.authenticate_reddit()

    @staticmethod
    async def authenticate_reddit():
        """Authenticates and returns a Reddit instance."""
        return asyncpraw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
        )

    def construct_query(self, location: str, interests: list[str]) -> str:
        """Constructs a query string from location and interests."""
        interests_part = " ".join(interests)
        return f"itinerary {location} {interests_part}"

    async def search_reddit_itineraries(self, query: str, subreddit="travel", max_results=5):
        """Searches Reddit for posts matching the query."""
        subreddit = await self.reddit.subreddit(subreddit)
        search_results = subreddit.search(query, limit=max_results)
        return [post async for post in search_results]

    @staticmethod
    def extract_text_from_reddit_post(post):
        """Extracts text from a Reddit post."""
        return post.selftext if post.selftext else post.title

    @staticmethod
    def summarize_text_with_llm(text: str, model_name="cnmoro/arcee-lite:q4_k_m"):
        """Summarizes text using the LLM with grammar constraints."""
        prompt = f"""
            Extract and summarize locations and key points from the text below. Provide the output as a list of objects with 'location' and 'description' fields:

            {text}
        """
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                format=LocationSummary.model_json_schema(),
                options={"temperature": 0.7},
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Error summarizing text: {e}")

    async def process_search_and_summarize(self, location: str, interests: list[str], subreddit="travel", max_results=5) -> str:
        """Searches Reddit and summarizes results."""
        query = self.construct_query(location, interests)
        posts = await self.search_reddit_itineraries(query, subreddit, max_results)

        all_summaries = []
        for post in posts:
            text = self.extract_text_from_reddit_post(post)
            if text:
                try:
                    summary_json = self.summarize_text_with_llm(text)
                    summary_dict = json.loads(summary_json)
                    summary = LocationSummary(**summary_dict)
                    all_summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing post '{post.title}': {e}")
            else:
                print(f"No text extracted from post '{post.title}'")

        combined_summaries = "\n".join(
            f"- {item.location}: {item.description}" for item in all_summaries
        )
        return combined_summaries