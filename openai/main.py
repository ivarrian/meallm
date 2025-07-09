from dotenv import load_dotenv
from agents.mcp import MCPServerStdio
from agents import Agent, Runner, trace
from IPython.display import display, Markdown
import asyncio
import os
from datetime import datetime
from contextlib import AsyncExitStack
from pydantic import BaseModel
from typing import List


# Load environment variables from .env file
load_dotenv(override=True)

# --- Configuration for all our MCP servers ---

# Public holiday server configuration
public_holiday_params = {"command": "uv", "args": ["run", "vic-au-dates-mcp-server"]}

# Todoist server configuration
todoist_params = {
    "command": "npx",
    "args": ["-y", "@kydycode/todoist-mcp-server-ext@latest"],
    "env": {"TODOIST_API_TOKEN": os.getenv("TODOIST_API_KEY")}
}

# Playwright server configuration
playwright_params = {"command": "npx", "args": ["-y", "@playwright/mcp@latest"]}

mcp_server_params = [public_holiday_params, playwright_params]


# class Recipe(BaseModel):
#     recipe_url: str
#     name_of_recipe: str
#     description: str
#     cook_time: str
#     total_time: str
#     recipe_category: str
#     recipe_ingredients: List[str]

class BaseIngredient(BaseModel):
    ingredient_name: str

class BaseIngredients(BaseModel):
    ingredients: List[BaseIngredient]

class PublicHolidays(BaseModel):
    monday: bool
    tuesday: bool
    wednesday: bool
    thursday: bool
    friday: bool

async def main():

    public_holiday_instructions = f"""
    You use the get_dates tools to determine what days of the week is a public holiday in Victoria, Australia.
    The current datetime is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    

    ingredient_extractor_instructions = f"""
    You receive input from the user for the types of recipes they would like to make for the week. 
    Extract the list of base ingredients from the request and return the output . After you have extracted the ingredients, you handoff to the Public Holiday Agent
"""

    # Use AsyncExitStack to manage the lifecycle of all servers
    async with AsyncExitStack() as stack:
        # For each server, call its connect() method, which returns
        # the context manager that we then enter.

        public_holiday_mcp_servers = [await stack.enter_async_context(MCPServerStdio(public_holiday_params))]
        public_holiday_agent = Agent(
            name="PublicHolidayAgent",
            instructions=public_holiday_instructions,
            model="gpt-4o-mini",
            mcp_servers = public_holiday_mcp_servers,
            output_type=PublicHolidays,
        )
        
        # mcp_servers = [await stack.enter_async_context(MCPServerStdio(params)) for params in mcp_server_params]
        # Now that all servers are connected, initialize the agent
        ingredient_extractor_agent = Agent(
            name="IngredientExtractor",
            instructions=ingredient_extractor_instructions,
            model="gpt-4o-mini",
            # mcp_servers=mcp_servers,
            output_type=BaseIngredients,
            handoffs=[public_holiday_agent]
        )

        # Define the user's request
        request = "I would like to make at least one vegetarian recipe and others could be with chicken or fish"

        # Trace the execution of the agent
        with trace(ingredient_extractor_agent.name):
            result = await Runner.run(ingredient_extractor_agent, request)
        


if __name__ == "__main__":
    # Run the main asynchronous function
    asyncio.run(main())