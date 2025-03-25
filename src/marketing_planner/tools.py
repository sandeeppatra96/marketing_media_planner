"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

import json
from typing import Any, Optional, cast

import aiohttp
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

from marketing_planner.configuration import Configuration
from marketing_planner.state import State
from marketing_planner.utils import init_model

async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query a search engine.

    This function queries the web to fetch comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events. Provide as much context in the query as needed to ensure high recall.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


_INFO_PROMPT = """You are doing web research on behalf of a user. You are trying to find out this information:

<info>
{info}
</info>

You just scraped the following website: {url}

Based on the website content below, jot down some notes about the website.

<Website content>
{content}
</Website content>"""


async def scrape_website(
    url: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Scrape and summarize content from a given URL.

    Returns:
        str: A summary of the scraped content, tailored to the extraction schema.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()

    p = _INFO_PROMPT.format(
        info=json.dumps(state.marketing_plan, indent=2),
        url=url,
        content=content[:40_000],
    )
    raw_model = init_model(config)
    result = await raw_model.ainvoke(p)
    return str(result.content)


async def analyze_website(
    url: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Analyze a business website to extract marketing-relevant information.
    
    This tool analyzes a business website to extract key marketing information
    like industry, products/services, target audience and existing marketing.
    
    Args:
        url: The business website URL to analyze.
    """
    # First scrape the website
    website_content = await scrape_website(url, state=state, config=config)
    
    # Now analyze it
    analysis_prompt = """You've just scraped a business website. Based on the content, analyze and extract:
    1. Industry/Niche
    2. Products/Services offered
    3. Target Audience (if identifiable)
    4. Existing Marketing Strategies (social media presence, content marketing, etc.)
    
    Website content:
    {content}
    
    Provide a concise analysis of these elements."""
    
    p = analysis_prompt.format(content=website_content)
    raw_model = init_model(config)
    result = await raw_model.ainvoke(p)
    
    # Note: We don't need custom memory tools anymore as we're using the library-provided ones
    return str(result.content)