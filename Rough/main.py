from fastapi import FastAPI
import uvicorn

app = FastAPI()

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

class Menu:

    # Start a basic Menu template 
    def __init__(self, id: int, type: enumerate, name: str, price: int):
        self.id = id 
        self.type = type
        self.name = name
        self.price = price

    # Add a new dish to the Menu
    def add_dish(self, name):
        

