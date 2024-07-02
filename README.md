# TaleTutors: Narrative-Based Teaching Voice Bot


## Inspiration
The inspiration for TaleTutor came from the desire to make learning more engaging and effective for students. We recognized that traditional educational methods often fail to connect topics holistically and relate concepts to real-world scenarios, leading to disinterest and fragmented understanding. Drawing from research on narrative learning and the transformative power of storytelling, we envisioned an AI-driven platform that uses immersive narratives to captivate students' imaginations. TaleTutor strives to bridge the gap between abstract concepts and practical applications, making education both fun and meaningful.

## What it does
Our narrative-based learning system serves three primary purposes: Relating material to real-world scenarios, fostering student comfort, and making learning enjoyable. It features a chatbot with dual interfaces for teachers and students. Teachers can upload lesson materials, while **students choose their learning theme, such as Harry Potter**** in our demonstration with subjects like Physics and Civics. 

Refer to the demonstration in this [YouTube video](https://youtu.be/jX9IUWnVgH0?si=5nRHhNS4EOxanqFl)

The interactive chat system, led by the character Dumbledore, narrates stories and explains concepts dynamically based on student interaction. It adeptly redirects off-topic inquiries back to the lesson material, simultaneously notifying teachers for follow-up. Through fictional characters and their worlds, we effectively teach real-world scenarios using engaging narratives.

## How we built it

![TaleTutor Router Architecture](https://github.com/Niranjan-Cholendiran/TaleTutors/assets/78549555/8faba4cb-57f1-493d-8bc4-6c609ea3b315)

The RAG application utilizes a LangChain-supported GPT-3.5 LLM. This model retrieves PDF uploaded by the teacher, converts it into a vector store to build a knowledge base, and performs RAG operations based on student queries.

Subsequently, the RAG output is processed by another GPT-3.5 LLM, which acts as a router. It analyzes student responses to decide whether to proceed with narrative building or query the LangChain-supported GPT-3.5 LLM for knowledge retrieval from the knowledge base. Once relevant knowledge is gathered, the GPT-3.5 LLM employs zero-shot prompting to generate narratives. The system also manages queries that are out of scope by gently guiding students back on topic or raising a ticket to the teacher for inquiries that are relevant but off-topic.

Additionally, we have developed a Python script that interfaces with our LLMs hosted on a Flask server. Using the Axios library, we establish connections to localhost to send and receive data. This script facilitates retrieving inputs and displaying outputs within the chatbot interface as required.

## Highlights 
* TaleTutor was selected as one of the top 10 projects in UC Berkeley's AI hackathon out of 290 projects.
* TaleTutor prevents students from going off-topic, avoids hallucinations, and creates tickets for teachers to address out-of-syllabus questions.
* TaleTutor accurately integrates lessons with movie stories, ensuring distinguishable custom voices for movie characters.

Find more information about TaleTutor [here](https://devpost.com/software/taletutor).
