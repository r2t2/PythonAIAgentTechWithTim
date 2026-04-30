from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]


# llm = ChatOpenAI(model="gpt-40-mini")
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
llm = ChatAnthropic(model="claude-sonnet-4-6")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
prompt2 = SystemMessagePromptTemplate.fr(
    [
        (
            "system",
            """
            """
        )
    ]
)

agent = create_agent(
    tools=[],
    system_prompt=prompt,
)
