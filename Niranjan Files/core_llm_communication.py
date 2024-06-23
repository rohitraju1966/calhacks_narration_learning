from langchain.agents import tool
from pydantic.v1 import BaseModel, Field
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
from text2speech import *
from core_knowledge_llm_RAG import *
# from langchain_anthropic import ChatAnthropic
import os 
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
#os.getenv("OPENAI_API_KEY")
#os.environ["OPENAI_API_KEY"] = ""


movie="Harry Potter"
subject= None # ["Physics", "civics"]
chapter= None # ["Newton's Laws of Motion", "Role of Political Parties"]
chapter_pdf_mapping= {
"Newton's Laws of Motion":r"C:\Users\saaij\OneDrive - UCB-O365\01. Study\04. Hackathon and Competitions\05. UC Berkely Hackathon 06-22-2024\02. Hacking\Final_Repo_2\calhacks_narration_learning\Niranjan Files\pdfs\Newtons_laws_NASA.pdf" 
"Role of Political Parties": r"C:\Users\saaij\OneDrive - UCB-O365\01. Study\04. Hackathon and Competitions\05. UC Berkely Hackathon 06-22-2024\02. Hacking\Final_Repo_2\calhacks_narration_learning\Niranjan Files\pdfs\Civics_Role_Of_Political_Parties.pdf"
} #CHANGED
initial_context= None

chat_history= None
vectorstore= None 
model= None
st_name= None

movie_mapping= {"Harry Potter": "Dumbledore"}

# Defining the knowledge_llm_func's input schema
class knowledge_llm_func_schema(BaseModel):
    user_query: str= Field(...,description= "Query asked by the user")


@tool(args_schema=knowledge_llm_func_schema)
def knowledge_llm_calling(user_query: str) -> str:
    """A function that can only answer questions on Physics subject on the Newton's Law of Motion chapter"""
    global subject, chapter, movie, initial_context, chat_history, vectorstore,chapter_pdf_mapping, model, st_name
    context= RAG_inference(question= user_query, vectorstore= vectorstore, model= model)
    return context

# Define the initial story model 
story_mode= ChatOpenAI(#model= ,
    openai_api_key= os.getenv("OPENAI_API_KEY"), 
    temperature=0, 
    #prompt=prompt
    )

def initial_user_input(st_name, subject_user, chapter_user, theme):
    global subject, chapter, movie, initial_context, chat_history, vectorstore,chapter_pdf_mapping, model, st_name
    subject= subject_user
    chapter= chapter_user
    st_name= st_name
    movie= theme
    vectorstore, model= model_embeddings_vectordatabase(pdf= chapter_pdf_mapping[chapter_user], key= os.getenv("OPENAI_API_KEY"))
    print("Vector Store Created")
    if(chapter=="Role of Political Parties"):
        question= f"In detail explain the Role of Political Parties: Power, Reporting, and Influence on Central Government Decisions?"
    else:
        question= f"In detail explain the idea of all three Newton's Law of Motion?" #CHANGED

    initial_context= RAG_inference(question=question, vectorstore= vectorstore, model= model,key= os.getenv("OPENAI_API_KEY"))
    #initial_context= knowledge_llm_calling({'user_query': "Explain in detail, what is {chapter}? ", 'subject': subject, 'chapter': chapter})
    prompt= ChatPromptTemplate.from_messages([
    ("user", 
        f"""
    Imagine you are {movie_mapping[movie]} from the {movie} movie and you are going to teach me (your student) about {chapter}. Use the below context to teach:

    Context: {initial_context}

    Your teaching style:
    * Teach concepts by narrating a story happening in the {movie} universe using characters from the movie.
    * Fore every new question or new concept you teach, try to continue the same story.
    * Use {movie_mapping[movie]}'s slang and narrate the story from {movie_mapping[movie]}'s POV.
    * Treat me as a 10 year old kid
    * Narrate a story and ask questions in between to guide me in the right path to learn the concepts. Occasionally, pause and conduct easy quiz to make sure I've understood the concepts correctly and  re-explain using if I'm wrong.
    * After asking a question, wait for my response and ONLY PROCEED AFTER GETTING THE RESPONSE.
    * Do not send more than 100 words in every iteration."""
        )
    ])
    chain_story_model= prompt | story_mode #| OpenAIFunctionsAgentOutputParser() | route_to_knowledge_llm    
    story_llm_response =chain_story_model.invoke({"chapter":chapter,"movie": movie,"initial_context":initial_context})
    chat_history = f"AI: {story_llm_response.content}"
    trasncript= story_llm_response.content
    print("trasncript:", trasncript)
    #audio_path= text_to_speech(trasncript)
    #print("audio_path:", audio_path)
    
   
#initial_user_input("Niranjan", "Physics", "Newton's Laws of Motion", "Harry Potter")


# For reply with 
story_mode1= ChatOpenAI(#model= ,
    openai_api_key= os.getenv("OPENAI_API_KEY"), 
    temperature=0, 
    #prompt=prompt
    ).bind_tools(tools= [knowledge_llm_calling])

prompt1= ChatPromptTemplate.from_messages([
    ("system", "If the user's input is a response to the your previous question in the mentioned chat history, continue the story. If the user has asked a new question from the {chapter}, ONLY THEN call the knowledge_llm_calling function. If the user's response is irrelevant to the subject and previous conversation, remind to get back to the topic and gently continue"),
    ("user", "Chat History is: {chat_history} User's input is:{user_input}")    
])

chain_story_model1= prompt1 | story_mode1 #| OpenAIFunctionsAgentOutputParser() | route_to_knowledge_llm

# For reply with knowledge LLM's result
story_mode2= ChatOpenAI(#model= ,
    openai_api_key= os.getenv("OPENAI_API_KEY"), 
    temperature=0, 
    )

prompt2= ChatPromptTemplate.from_messages([
    ("system", "You have been teaching the user in {movie_mapping[movie]}'s slang and the user has asked you the below provided question. See if the below provided context can be used to answer the question. If yes, answer it. BUT ONLY IF THE CONTEXT IS NOT SUFFICIENT, say the teacher will address that question and continue the previous conversation."),
    ("user", "Chat History is: {chat_history} \n\n User's question is: {user_input} \n\n Context is: {new_content}")    
])

chain_story_model2= prompt2 | story_mode2


# Function call for the subsequent responses
def chat_user_input(user_input):
    global subject, chapter, movie, initial_context, chat_history, vectorstore,chapter_pdf_mapping, model, st_name
    # Inference 1
    story_llm_response1 =chain_story_model1.invoke({'user_input':user_input , "chapter":chapter,"subject":subject,"chat_history":chat_history})
    #story_llm_response =chain.invoke({'user_input': "I know this concept, can you teach me what's Newton's Third Law please?", "chapter":chapter,"subject":subject,"chat_history":chat_history })
    #story_llm_response = chain.invoke({'user_input': "Maybe Harry stay in the same place?", "chapter":chapter,"subject":subject,"chat_history":chat_history })

    if (story_llm_response1.content==''):
        # Call the knowledge model function
        function_name= story_llm_response1.additional_kwargs['tool_calls'][0]['function']['name']
        function_args= json.loads(story_llm_response1.additional_kwargs['tool_calls'][0]['function']['arguments'])
        #print(function_args)
        new_content= knowledge_llm_calling(function_args)
        story_llm_response2 =chain_story_model2.invoke({'user_input':user_input , 'new_content': new_content, "chapter":chapter,"subject":subject,"chat_history":chat_history,'movie':movie})
        chat_history += f"\n\n Human:{user_input} \n\nAI: {story_llm_response2.content}"
        trasncript= story_llm_response2.content
        
        # Create teacher ticket
        if ("teacher" in trasncript):
            ticket_data= {"STUDENT_NAME": st_name,
             "TOPIC": subject,
             "CHAPTER": chapter,
             "QUESTION":chat_history,
             "CONVERSATION_HISTORY": chat_history,
             "STATUS": "Unresolved"
            }

            # public\teacher_question_ticket.xlsx does not exist, create an excel with ticket_data
            # Else 
            ticket_df= pd.read_csv()
            file_path= r'public\teacher_question_ticket.xlsx'
            # Add ticket_data as a new row and 
            if not os.path.exists(file_path):
                ticket_df = pd.DataFrame([ticket_data])
                ticket_df.to_excel(file_path, index=False)
            else:
                ticket_df = pd.read_excel(file_path)
                ticket_df = ticket_df.append(ticket_data, ignore_index=True)
                ticket_df.to_excel(file_path, index=False)            
        #audio_path= text_to_speech(trasncript)
        #print("audio_path:", audio_path)
        print("trasncript:", trasncript)
    else:
        # Update the chat history
        chat_history = str(chat_history) + f"\n\n Human:{user_input} \n\nAI: {story_llm_response1.content}"
        trasncript= story_llm_response1.content
        #audio_path= text_to_speech(trasncript)
        #print("audio_path:", audio_path)
        print("trasncript:", trasncript)