"""
This module contains classes for abstract questions.

The QuestionsExtractor class uses Generative AI to extract questions. It takes conversation with My-GPTs as an input and starts the process of extracting questions and store it in new json file.


## Example:
    ```python
    >>> from questions_extractor import QuestionsExtractor
    >>> extractor = QuestionsExtractor('111403538')
    >>> extractor.start_process()
    ```
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

class QuestionsExtractor:
    """
    Extracts questions using Generative AI.

    ## Attributes:
        student_id (str): The ID of the student.

    ## Methods:
        start_process: Starts the process of extracting questions and create new json file.

    ## Example:
    ```python
    >>> extractor = QuestionsExtractor('111403538')
    >>> extractor.start_process()
    ```
    """

    def __init__(self, student_id: str) -> None:
        """
        Initializes the QuestionsExtractor object.
        """
        self.student_id = student_id
        self.user_prompt_template = self._initialize_user_prompt_template()
        self.system_prompt_template = self._initialize_system_prompt_template()

        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def _initialize_system_prompt_template(self) -> str:
        """
        Initializes the system prompt template.

        Returns:
            SystemTemplate: The initialized system prompt template.
        """
        system_template = """
# 背景設定

1. 你是一位對繁體中文有很深造詣的文學家，你很擅長針對問題進行總結。

2. 你不會解答問題。

3. 你會收到多個問題標題與每個問題標題的問題，其中問題以列點的方式展示，有幾個問題標題你就要有幾個問題的整理區塊，你要針對所有問題去進行同質性整合，將問題整理並整合成一個100字到300字以內的摘要。

# 回應格式
{
"問題標題": "問題總結",
"問題標題": "問題總結",
"問題標題": "問題總結"
}
"""

        return system_template

    def _initialize_user_prompt_template(self) -> str:
        """
        Initializes the user prompt template.

        Returns:
            UserTemplate: The initialized user prompt template.
        """
        mygpt = self._get_mygpt()

        user_template = f"""
# 問題
{mygpt}
"""

        return user_template

    def _get_mygpt(self) -> str:
        """
        Get the conversation with My-GPTs.

        Returns:
            str: The conversation with My-GPTs.
        """
        result = ""

        with open(f"mygpt/{self.student_id}.json", "r", encoding='utf-8') as file:
            mygpt = json.load(file)

            for key, value in mygpt.items():
                result += "# 問題標題"
                result += f"{key}:\n\n"
                result += "# 問題\n"

                for idx, question in enumerate(value, start=1):
                    result += f"{idx}. {question}\n"

                result += "\n"

        return result

    def _get_questions(self, model: str) -> str:
        """
        Get the questions extracted by Generative AI.
        
        Args:
            model (str): The model to use for extracting questions.

        Returns:
            str: The extracted questions.
        """
        messages = [
            {"role": "system", "content": self.system_prompt_template},
            {"role": "user", "content": self.user_prompt_template}
        ]

        chat_completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.6,
        )

        return chat_completion.choices[0].message.content

    def start_process(self) -> str:
        """
        Starts the process of extracting questions and store it in new json file.

        Returns:
            str: The extracted questions.
        """
        model = "gpt-3.5-turbo"
        result = self._get_questions(model)

        if not os.path.exists("questions"):
            os.makedirs("questions")

        with open(f"questions/{self.student_id}.json", "w", encoding='utf-8') as file:
            json.dump(json.loads(result), file, ensure_ascii=False, indent=4)
