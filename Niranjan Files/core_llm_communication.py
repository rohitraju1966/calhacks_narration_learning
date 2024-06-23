from langchain.agents import tool
from pydantic.v1 import BaseModel, Field
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
# from langchain_anthropic import ChatAnthropic
import os 
from dotenv import load_dotenv
load_dotenv()
#os.getenv("OPENAI_API_KEY")



# Defining the knowledge_llm_func's input schema
class knowledge_llm_func_schema(BaseModel):
    user_query: str= Field(...,description= "Query asked by the user")
    subject: str= Field(...,description="Subject name")
    chapter: str= Field(...,description="Chapter name")


@tool(args_schema=knowledge_llm_func_schema)
def knowledge_llm_calling(user_query: str, subject= subject, chapter= chapter) -> str:
# def knowledge_llm_calling(user_query: str, subject_chapter_tuple: list) -> str:
    """A function that can only answer questions on Physics subject on the Newton's Law of Motion chapter"""
    return "TEST"


def audio_conver_and_return(text):
    print(text)
    print("audio file converted and sent")
    return "audio file converted and sent"


movie="Harry Potter"
chapter= None
initial_context= None
subject= None
chat_history= None

movie_mapping= {"Harry Potter": "Dumbledore"}

# For reply with 
story_mode= ChatOpenAI(#model= ,
    openai_api_key= os.getenv("OPENAI_API_KEY"), 
    temperature=0, 
    #prompt=prompt
    )

prompt= ChatPromptTemplate.from_messages([
    ("user", 
     f"""
Imagine you are {movie_mapping[movie]} from the {movie} movie and you are going to teach me (your student) about {chapter}. Use the below context to teach:

Context: {initial_context}

Your teaching style:
* Teach concepts by narrating a story happening in the Harry Potter universe using characters from the movie.
* Use Dumbledore's slang and narrate the story from Dumbledore's POV.
* Treat me as a 10 year old kid
* Narrate a story and ask questions in between to guide me in the right path to learn the concepts. Occasionally, pause and conduct easy quiz to make sure I've understood the concepts correctly and  re-explain using if I'm wrong.
* After asking a question, wait for my response and only then proceed to continue teaching."""
     )
])

chain_story_model= prompt | story_mode #| OpenAIFunctionsAgentOutputParser() | route_to_knowledge_llm


def initial_user_input(st_name, subject_user, chapter_user, theme):
    subject= subject_user
    chapter= chapter_user
    movie= theme
    initial_context= knowledge_llm_calling({'user_query': "Explain in detail, what is {chapter}? ", 'subject': subject, 'chapter': chapter})
    story_llm_response =chain_story_model.invoke({"chapter":chapter,"movie": movie,"initial_context":initial_context})
    chat_history = "AI: {story_llm_response1.content}"
    audio_conver_and_return(story_llm_response.content)
   

#initial_user_input("Niranjan", "Physics", "Newton's Law of Motion", "Harry Potter")


# For reply with 
story_mode1= ChatOpenAI(#model= ,
    openai_api_key= os.getenv("OPENAI_API_KEY"), 
    temperature=0, 
    #prompt=prompt
    ).bind_tools(tools= [knowledge_llm_calling])

prompt1= ChatPromptTemplate.from_messages([
    ("system", "If the user's input is a response to the your previous question in the mentioned chat history, continue the story. If the user has asked a new question from the Newton's Law of Function, ONLY THEN call the knowledge_llm_calling function. If the user's response is irrelevant to the subject and previous conversation, remind to get back to the topic and gently continue"),
    ("user", "Chat History is: {chat_history} User's input is:{user_input}")    
])

chain_story_model1= prompt1 | story_mode1 #| OpenAIFunctionsAgentOutputParser() | route_to_knowledge_llm

# For reply with knowledge LLM's result
story_mode2= ChatOpenAI(#model= ,
    openai_api_key= os.getenv("OPENAI_API_KEY"), 
    temperature=0, 
    )

prompt2= ChatPromptTemplate.from_messages([
    ("system", "You have been teaching the user in Dumbledoor's slang and the user has asked you the below provided question. See if the below provided context can be used to answer the question. If yes, answer it. If NOT, say the teacher will address that question and continue the previous conversation."),
    ("user", "Chat History is: {chat_history} \n\n User's question is: {user_input} \n\n Context is: {new_content}")    
])

chain_story_model2= prompt2 | story_mode2

# Function call
def chat_user_input(user_input):
    # Inference 1
    user_input= "I know this, can you teach me Newton's second law of motion?"
    story_llm_response1 =chain_story_model1.invoke({'user_input':user_input , "chapter":chapter,"subject":subject,"chat_history":chat_history})
    #story_llm_response =chain.invoke({'user_input': "I know this concept, can you teach me what's Newton's Third Law please?", "chapter":chapter,"subject":subject,"chat_history":chat_history })
    #story_llm_response = chain.invoke({'user_input': "Maybe Harry stay in the same place?", "chapter":chapter,"subject":subject,"chat_history":chat_history })

    if (story_llm_response1.content==''):
        # Call the knowledge model function
        function_name= story_llm_response1.additional_kwargs['tool_calls'][0]['function']['name']
        function_args= json.loads(story_llm_response1.additional_kwargs['tool_calls'][0]['function']['arguments'])
        #print(function_args)
        new_content= knowledge_llm_calling(function_args)
        story_llm_response2 =chain_story_model2.invoke({'user_input':user_input , 'new_content': new_content, "chapter":chapter,"subject":subject,"chat_history":chat_history })
        chat_history += "\n\n Human:{user_input} \n\nAI: {story_llm_response2.content}"
        audio_conver_and_return(story_llm_response2.content)
    else:
        # Update the chat history
        chat_history += "\n\n Human:{user_input} \n\nAI: {story_llm_response1.content}"
        audio_conver_and_return(story_llm_response1.content)