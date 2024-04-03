from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os

if __name__ == "__main__":
    load_dotenv()

    information = """
        Tigranes II, more commonly known as Tigranes the Great (Tigran Mets in Armenian;[4][a] Ancient Greek: Τιγράνης ὁ Μέγας, Tigránes ho Mégas; Latin: Tigranes Magnus;[6] 140 – 55 BC), was a king of Armenia. A member of the Artaxiad dynasty, he ruled from 95 BC to 55 BC. Under his reign, the Armenian kingdom expanded beyond its traditional boundaries and reached its peak, allowing Tigranes to claim the title Great King or King of Kings. His empire for a short time was the most powerful state to the east of the Roman Republic. Tigranes's title King of Kings is linked, along his victories, also to the appearance of Halley comet during his reign [7], as depicted on the rare series of Tigranes's coins. [8] Either the son or nephew of Artavasdes I, Tigranes was given as a hostage to Mithridates II of Parthia after Armenia came under Parthian suzerainty. After ascending to the Armenian throne, he rapidly expanded his kingdom by invading or annexing Roman and Parthian client-kingdoms. Tigran decided to ally with Mithridates VI of Pontus by marrying his daughter Cleopatra. At its height, Tigranes' empire stretched from the Pontic Alps to Mesopotamia and from the Caspian Sea to the Mediterranean. With captured vassals, he had even reached the Red Sea and the Persian Gulf. Many of the inhabitants of conquered cities were forcibly relocated to his new capital, Tigranocerta. An admirer of the Greek culture, Tigranes invited many Greek rhetoricians and philosophers to his court, and his capital was noted for its Hellenistic architecture. Armenia came into direct conflict with Rome after Mithridates VI was forced to seek refuge in Tigranes' court. In 69 BC, Tigranes was decisively defeated at the Battle of Tigranocerta by a Roman army under the command of Lucullus, and a year later he met another major defeat at Artaxata, the old Armenian capital. The recall of Lucullus gave Tigranes a brief respite, but in 66 BC Armenia faced another Roman invasion led by Pompey, aided by Tigranes' own son, Tigranes the Younger. Tigranes chose to surrender and was allowed to retain the heartland of his kingdom as a Roman buffer state, while all of his conquests were annexed. He continued to rule Armenia as a formal ally of Rome until his death around 55 BC at the age of 85.    
    """
    summary_template = """
        given the {information} about a person I want you to create
        1. A short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    res = chain.invoke(input={"information": information})
