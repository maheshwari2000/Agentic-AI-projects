import requests
from bs4 import BeautifulSoup
from typing import TypedDict
from langchain.agents import tool
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# -------------------------
# STEP 1: Define State
# -------------------------
class CityDetails(TypedDict):
    input: str
    city1: str
    city2: str
    cOfLoving1: str
    cOfLoving2: str
    qOfLoving1: str
    qOfLoving2: str
    result: str

# -------------------------
# STEP 2: Define Tools
# -------------------------
@tool
def get_cost_of_living_info(location: str) -> str:
    """Fetches cost of living information for a given location."""
    try:
        url = f"https://www.numbeo.com/cost-of-living/in/{location.replace(' ', '-')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        div = soup.find(
            'div', 
            {'class': 'seeding-call table_color summary limit_size_ad_right padding_lower other_highlight_color'}
        )
        return div.text.strip() if div else "Cost of living data not found."
    except Exception as e:
        return f"Error fetching data for {location}: {str(e)}"


@tool
def get_quality_of_living_info(location: str) -> str:
    """Fetches quality of living information for a given location."""
    location = location.replace(" ", "-")
    quality_of_living = {
        "New-York": "High competition, high salary, best for finance & tech.",
        "San-Francisco": "Tech hub, many startups, expensive but high salaries.",
        "Austin": "Growing tech scene, affordable living.",
        "Berlin": "Strong finance, healthcare, and consulting job market."
    }
    return quality_of_living.get(location, "Quality of living data not available.")


@tool
def get_property_info(location: str) -> str:
    """Fetches property investment information for a given location."""
    try:
        url = f"https://www.numbeo.com/property-investment/in/{location.replace(' ', '-')}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find('table', class_="table_indices")
        rows = table.find_all('tr')
        data = f"Property Investment Data for {location}"
        for i in range(1, len(rows)):
            tds = rows[i].find_all('td')
            td_data = tds[0].text + tds[1].text.replace("\n", "")
            data += "; " + td_data
        return data
    except:
        return f"Property investment data not found for {location}."

# -------------------------
# STEP 3: Setup Tools & LLM
# -------------------------
tools = [get_cost_of_living_info, get_quality_of_living_info, get_property_info]
tool_node = ToolNode(tools)
llm = ChatGroq(model="llama3-8b-8192", verbose=True)
llm_with_tools = llm.bind_tools(tools)

# -------------------------
# STEP 4: Decision Assistant
# -------------------------
def decisionAssistant(state: CityDetails) -> CityDetails:
    messages = [
        SystemMessage(content="""
        You are an intelligent career decision specialist and have been provided with tools to compare:
        Cost of Living, Quality of Living, and Property Investment Data for two locations.
        
        Based on the tool results, analyze and give a **final comparison**:
            - Cost of living comparison
            - Quality of living comparison
            - Property investment comparison
            - Final conclusion: Which city should I choose and why?
        """),
        HumanMessage(content=state['input'])
    ]
    state['result'] = llm_with_tools.invoke(messages).content
    return state

# -------------------------
# STEP 5: Build Graph
# -------------------------
graph = StateGraph(CityDetails)

graph.add_node('decisionAssistant', decisionAssistant)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'decisionAssistant')
graph.add_conditional_edges('decisionAssistant', tools_condition)
graph.add_edge('tools', 'decisionAssistant')

agent = graph.compile()

# -------------------------
# STEP 6: Run Agent
# -------------------------
result = agent.invoke({
    "input": "I am currently a Data Scientist in Austin, and I have a new offer for a Senior Data Scientist position in Berlin. Should I take it?",
    "city1": "Austin",
    "city2": "Berlin",
    "cOfLoving1": "",
    "cOfLoving2": "",
    "qOfLoving1": "",
    "qOfLoving2": "",
    "result": ""
})

print(result["result"])
