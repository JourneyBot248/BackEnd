import json
import time
import ollama
from pydantic import BaseModel, Field
from typing import List
import requests
from requests.structures import CaseInsensitiveDict
from dotenv import load_dotenv
import os
from reddit_scraper import RedditSummarizer 

load_dotenv()

class Activity(BaseModel):
    location_name: str
    description: str = Field(..., min_length=50)
    longitude: float = None
    latitude: float = None

class Day(BaseModel):
    day: int
    schedule: List[Activity]

class Itinerary(BaseModel):
    destination: str
    trip_duration: int
    itinerary: List[Day] 

def geocode_location(location_name):
    """Uses the Geoapify API to retrieve latitude and longitude for a given location name."""
    url = f"https://api.geoapify.com/v1/geocode/search?text={location_name}&apiKey={os.getenv('GEOAPIFY_API_KEY')}"
    headers = CaseInsensitiveDict({"Accept": "application/json"})
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("features"):
                coords = data["features"][0]["geometry"]["coordinates"]
                return {"longitude": coords[0], "latitude": coords[1]}
            else:
                raise ValueError(f"No geocoding results found for {location_name}")
        else:
            raise RuntimeError(f"Geoapify API error: {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"Error during geocoding: {e}")

def generate_itinerary(destination, duration, interests, additional_info):
    """ Generates a detailed itinerary based on the destination, duration, interests, and additional information. """
    
    prompt = f"""
        Generate an engaging and highly-detailed itinerary for a trip to {destination} for {duration} days.
        Focus on including a mix of activities that align with these specific interests: {', '.join(interests)}. Each day should include around 4 ~ 5 activities.
        Make sure activities are diverse (e.g., museums, hands-on experiences, unique dining spots) and appealing to someone passionate about {', '.join(interests)}.
        
        For each activity, include:
          - Location name (specific location address for the activity, such as a specific store, landmark, restaurant, beach, or venue)
          - Detailed description (at least 50 characters) with specific recommendations
                
        Ensure the itinerary contains exactly {duration} days and the activities do not overlap with one another.
        
        Here is an example of the expected JSON format:

        Example 1:
        {{
          "destination": "Japan",
          "duration": 2,
          "itinerary": [
            {{
              "day": 1,
              "schedule": [
                {{
                  "location_name": "Tsukiji Outer Market",
                  "description": "Start your day with a visit to the famous Tsukiji Outer Market. Try fresh sushi for breakfast at Sushi Dai, known for its excellent omakase.",
              
                }},
                {{
                  "location_name": "Ueno Park",
                  "description": "Explore Ueno Park, home to several museums. Visit the Tokyo National Museum to learn about Japanese art and history.",
                  
                }},
                {{
                  "location_name": "Ameyoko Shopping Street",
                  "description": "Experience the lively atmosphere of Ameyoko. Have lunch at Okonomiyaki Sometaro, a local favorite for Japanese savory pancakes.",
                  
                }},
                {{
                  "location_name": "Takeshita Street, Harajuku",
                  "description": "Dive into Tokyo's youth culture on Takeshita Street. Shop for unique fashion items and try rainbow cotton candy.",
                  
                }},
                {{
                  "location_name": "Shibuya Crossing",
                  "description": "Experience the famous Shibuya Crossing and visit the Hachiko statue. End your day with dinner at Ichiran Ramen Shibuya for delicious tonkotsu ramen.",
                  
                }}
              "day": 2,
              "schedule": [
                {{
                  "location_name": "Kiyomizu-dera Temple",
                  "description": "Start your Kyoto journey at this UNESCO World Heritage site. Enjoy the panoramic views of Kyoto from the temple's wooden terrace.",
                  
                }},
                {{
                  "location_name": "Ginkaku-ji (Silver Pavilion)",
                  "description": "Explore this Zen temple known for its beautiful sand garden and moss garden. Take a stroll along the Philosopher's Path.",
                  
                }},
                {{
                  "location_name": "Nanzenji Temple",
                  "description": "Visit this important Zen temple complex. Don't miss the massive San-mon gate and the unique aqueduct on the temple grounds.",
                  
                }},
                {{
                  "location_name": "Nishiki Market",
                  "description": "Explore Kyoto's famous food market. Have a late lunch trying various local delicacies like tako tamago and Kyoto-style sushi at Nishiki Sushi.",
                  
                }},
                {{
                  "location_name": "Pontocho Alley",
                  "description": "End your day with a dinner at Izusen for traditional Kyoto cuisine (kaiseki) while enjoying the atmospheric narrow alley lined with traditional buildings.",
                  
                }}
              ]
            }}
          ]
        }}

    Return the result as a JSON object with the following structure:
    {{
        "destination": "{destination}",
        "duration": {duration},
        "itinerary": [
            {{
                "day": 1,
                "schedule": [
                    {{
                        "location_name": str,
                        "description": str,
                      
                    }},
                    ...
                  continue this format for each of the days in the schedule.
                ]
            }},
            ...
        ]
    }}
    Ensure that the itinerary contains exactly {duration} days.

    Here's some information you should keep in mind from other travelers for the itinerary generation: 
    {additional_info}
    """ 

    # model_name = 'cnmoro/arcee-lite:q4_k_m'
    model_name = 'technobyte/c4ai-command-r7b-12-2024:Q5_K_M'
    try:
        # NOT TESTED for docker ollama, otherwise install ollama and run before running this code

        # url = "http://localhost:11434/api/v1/generate" 
        # headers = {"Content-Type": "application/json"}
        # payload = {
        #     "model": "technobyte/c4ai-command-r7b-12-2024:Q5_K_M",
        #     "prompt": prompt,
        #     "format": Itinerary.model_json_schema(),
        #     "options": {"temperature": 0.7}
        # }

        # response = requests.post(url, headers=headers, json=payload)
        # response.raise_for_status()
        # content = response.json()

        # itinerary_json = content.get("output")
        # itinerary = Itinerary.model_validate_json(itinerary_json)

        ollama.pull(model_name)
        print("Model pulled")

        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            format=Itinerary.model_json_schema(),
            options={"temperature": 0.7}
        )   
        print("response", response['message']['content'])
        itinerary = Itinerary.model_validate_json(response['message']['content'])
        
        for day in itinerary.itinerary:
            for activity in day.schedule:
                coords = geocode_location(activity.location_name)
                activity.longitude = coords["longitude"]
                activity.latitude = coords["latitude"]
        
        return itinerary.model_dump()
    
    except Exception as e:
        raise RuntimeError(f"Error generating itinerary: {e}")

def save_itinerary_to_file(itinerary, filename):
    try:
        with open(filename, 'w') as file:
            json.dump(itinerary, file, indent=2)
        print(f"Itinerary saved to {filename}")
    except Exception as e:
        raise RuntimeError(f"Error saving itinerary to file: {e}")

def process_reddit_and_generate_itinerary(destination, duration, interests):
    reddit = RedditSummarizer()
    reddit_additional_info = reddit.process_search_and_summarize(
        location=destination,
        interests=interests
    )
    return generate_itinerary(destination, duration, interests, reddit_additional_info)


# example usecase
# def main():
#     case = {
#         "destination": "Japan",
#         "duration": 5,
#         "interests": ["technology", "history", "food"]
#     }
    
#     start_time = time.time()
#     itinerary = process_reddit_and_generate_itinerary(
#         destination=case["destination"],
#         duration=case["duration"],
#         interests=case["interests"]
#     )
    
#     itinerary_time = time.time() - start_time
#     print(f"Itinerary generation completed in {itinerary_time:.2f} seconds.")
    
#     filename = f"{case['destination'].replace(' ', '_').lower()}_itinerary.json"
#     save_itinerary_to_file(itinerary, filename)

# if __name__ == "__main__":
#     main()


"""
1. user talks to chatbot (frontend <--> backend (keep track of user/chatbot chat history and send that + new message) <--> LLM (chatbot)  (chat history))
2. when user clicks finish button, LLM summarizes(?) chat history into a JSON object (frontend <--> backend calls LLM function to summarize <--> LLM save as variable)
3. LLM generates itinerary based on JSON object (LLM generate json object or variable --> backend --> frontend to render as itinerary on the web page)
4. if user wants to change itinerary, they can talk to chatbot again (frontend talks to chatbot --> backend to different endpoint --> LLM with different 
prompt to "fix" itinerary based on feedback)
"""
