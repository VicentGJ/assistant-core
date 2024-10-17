from dotenv import load_dotenv
from fastapi import FastAPI

from api.router import search, vectorization

load_dotenv()

app = FastAPI()
app.include_router(vectorization.router)
app.include_router(search.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
