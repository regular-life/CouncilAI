from fastapi import APIRouter
from pydantic import BaseModel
from app.retrieval.search_agent import WebSearchAgent

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    max_results: int = 3

@router.post("/search")
async def web_search(req: SearchRequest):
    agent = WebSearchAgent()
    results = agent.search_and_scrape(req.query, req.max_results)
    return {"results": results}
