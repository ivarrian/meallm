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

mcp_server_params = [public_holiday_params, todoist_params, playwright_params]


class Recipe(BaseModel):
    name_of_recipe: str
    description: str
    cook_time: str
    total_time: str
    recipe_category: str
    recipe_ingredients: List[str]


class RecipesOutput(BaseModel):
    recipes: List[Recipe]


async def main():
    """
    Main asynchronous function to run the meal planning agent.
    """
    instructions = f"""
    You receive input from the user for the types of recipes they would like to make for the week. Generate a recipe per working day of the week, do not generate a recipe for a public holiday in Victoria, Australia. 
    You have internet access and all recipes must be from the following websites : 
    https://hot-thai-kitchen.com/ 

The current datetime is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    # Use AsyncExitStack to manage the lifecycle of all servers
    async with AsyncExitStack() as stack:
        # For each server, call its connect() method, which returns
        # the context manager that we then enter.
        
        mcp_servers = [await stack.enter_async_context(MCPServerStdio(params)) for params in mcp_server_params]
        # Now that all servers are connected, initialize the agent
        meal_planner = Agent(
            name="MealPlanner",
            instructions=instructions,
            model="gpt-4o-mini",
            mcp_servers=mcp_servers,
            output_type=RecipesOutput,
        )

        # Define the user's request
        request = "I would like to make recipes this week that use chicken, beef and pork and gochujang"

        # Trace the execution of the agent
        with trace(meal_planner.name):
            result = await Runner.run(meal_planner, request)

        # Print the final output
        print("--- Agent's Final Output ---")
        print(result.final_output.model_dump_json)

        


if __name__ == "__main__":
    # Run the main asynchronous function
    asyncio.run(main())