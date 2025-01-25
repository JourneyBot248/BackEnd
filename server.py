import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from ollam import (
    process_reddit_and_generate_itinerary,
    save_itinerary_to_file,
    Itinerary,
)

app = FastAPI()

PORT = 8001

class ItineraryRequest(BaseModel):
    destination: str
    duration: int
    interests: List[str]

class SaveRequest(BaseModel):
    itinerary: Itinerary
    filename: str

@app.post("/generate-itinerary/")
async def generate_itinerary(request: ItineraryRequest):
    """
    Generate an itinerary based on the provided destination, duration, and interests.
    """
    try:
        itinerary = await process_reddit_and_generate_itinerary(
            destination=request.destination,
            duration=request.duration,
            interests=request.interests,
        )
        return itinerary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating itinerary: {str(e)}")

@app.post("/save-itinerary/")
def save_itinerary(request: SaveRequest):
    """
    Save the provided itinerary to a file with the given filename.
    """
    try:
        save_itinerary_to_file(request.itinerary, request.filename)
        return {"message": f"Itinerary saved successfully to {request.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving itinerary: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Itinerary API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

