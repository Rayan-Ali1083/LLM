from langchain_core.prompts import PromptTemplate

# Prompt template

template = PromptTemplate(
    template="""
        Provide {info_length} unique and interesting facts about the cat species '{cats_input}'. 
        Ensure the facts are well-researched, diverse.
        Keep the facts concise but informative under 25 words.
    """,
    input_variables=['cats_input', 'info_length'],
    validate_template = True
)

template.save('template.json')