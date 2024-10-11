from collections import OrderedDict
import concurrent.futures
import json
import logging
import os
import pickle
import re
import sys
from typing import List, Dict, Any

import httpx
import pandas as pd
import toml
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from trafilatura import extract

logging.getLogger("httpx").setLevel(logging.WARNING)  # Disable INFO logging for httpx.


def truncate_filename(filename, max_length=125):
    """Truncate filename to max_length to ensure the filename won't exceed the file system limit.

    Args:
        filename: str
        max_length: int, default to 125 (usual path length limit is 255 chars)
    """

    if len(filename) > max_length:
        truncated_filename = filename[:max_length]
        logging.warning(f"Filename is too long. Filename is truncated to {truncated_filename}.")
        return truncated_filename

    return filename


def load_api_key(toml_file_path):
    try:
        with open(toml_file_path, "r") as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"File not found: {toml_file_path}", file=sys.stderr)
        return
    except toml.TomlDecodeError:
        print(f"Error decoding TOML file: {toml_file_path}", file=sys.stderr)
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)


def makeStringRed(message):
    return f"\033[91m {message}\033[00m"


class QdrantVectorStoreManager:
    """
    Helper class for managing the Qdrant vector store, can be used with `VectorRM` in rm.py.

    Before you initialize `VectorRM`, call `create_or_update_vector_store` to create or update the vector store.
    Once you have the vector store, you can initialize `VectorRM` with the vector store path or the Qdrant server URL.
    """

    @staticmethod
    def _check_create_collection(client: QdrantClient, collection_name: str, model: HuggingFaceEmbeddings):
        """Check if the Qdrant collection exists and create it if it does not."""
        if client is None:
            raise ValueError("Qdrant client is not initialized.")
        if client.collection_exists(collection_name=f"{collection_name}"):
            print(f"Collection {collection_name} exists. Loading the collection...")
            return Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=model,
            )
        else:
            print(f"Collection {collection_name} does not exist. Creating the collection...")
            # create the collection
            client.create_collection(
                collection_name=f"{collection_name}",
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
            return Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=model,
            )

    @staticmethod
    def _init_online_vector_db(url: str, api_key: str, collection_name: str, model: HuggingFaceEmbeddings):
        """Initialize the Qdrant client that is connected to an online vector store with the given URL and API key.

        Args:
            url (str): URL of the Qdrant server.
            api_key (str): API key for the Qdrant server.
        """
        if api_key is None:
            if not os.getenv("QDRANT_API_KEY"):
                raise ValueError("Please provide an api key.")
            api_key = os.getenv("QDRANT_API_KEY")
        if url is None:
            raise ValueError("Please provide a url for the Qdrant server.")

        try:
            client = QdrantClient(url=url, api_key=api_key)
            return QdrantVectorStoreManager._check_create_collection(
                client=client, collection_name=collection_name, model=model
            )
        except Exception as e:
            raise ValueError(f"Error occurs when connecting to the server: {e}")

    @staticmethod
    def _init_offline_vector_db(vector_store_path: str, collection_name: str, model: HuggingFaceEmbeddings):
        """Initialize the Qdrant client that is connected to an offline vector store with the given vector store folder path.

        Args:
            vector_store_path (str): Path to the vector store.
        """
        if vector_store_path is None:
            raise ValueError("Please provide a folder path.")

        try:
            client = QdrantClient(path=vector_store_path)
            return QdrantVectorStoreManager._check_create_collection(
                client=client, collection_name=collection_name, model=model
            )
        except Exception as e:
            raise ValueError(f"Error occurs when loading the vector store: {e}")

    @staticmethod
    def create_or_update_vector_store(
        collection_name: str,
        vector_db_mode: str,
        file_path: str,
        content_column: str,
        title_column: str = "title",
        url_column: str = "url",
        desc_column: str = "description",
        batch_size: int = 64,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        vector_store_path: str = None,
        url: str = None,
        qdrant_api_key: str = None,
        embedding_model: str = "BAAI/bge-m3",
        device: str = "mps",
    ):
        """
        Takes a CSV file and adds each row in the CSV file to the Qdrant collection.

        This function expects each row of the CSV file as a document.
        The CSV file should have columns for "content", "title", "URL", and "description".

        Args:
            collection_name: Name of the Qdrant collection.
            vector_store_path (str): Path to the directory where the vector store is stored or will be stored.
            vector_db_mode (str): Mode of the Qdrant vector store (offline or online).
            file_path (str): Path to the CSV file.
            content_column (str): Name of the column containing the content.
            title_column (str): Name of the column containing the title. Default is "title".
            url_column (str): Name of the column containing the URL. Default is "url".
            desc_column (str): Name of the column containing the description. Default is "description".
            batch_size (int): Batch size for adding documents to the collection.
            chunk_size: Size of each chunk if you need to build the vector store from documents.
            chunk_overlap: Overlap between chunks if you need to build the vector store from documents.
            embedding_model: Name of the Hugging Face embedding model.
            device: Device to run the embeddings model on, can be "mps", "cuda", "cpu".
            qdrant_api_key: API key for the Qdrant server (Only required if the Qdrant server is online).
        """
        # check if the collection name is provided
        if collection_name is None:
            raise ValueError("Please provide a collection name.")

        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}
        model = HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        if file_path is None:
            raise ValueError("Please provide a file path.")
        # check if the file is a csv file
        if not file_path.endswith(".csv"):
            raise ValueError(f"Not valid file format. Please provide a csv file.")
        if content_column is None:
            raise ValueError("Please provide the name of the content column.")
        if url_column is None:
            raise ValueError("Please provide the name of the url column.")

        # try to initialize the Qdrant client
        qdrant = None
        if vector_db_mode == "online":
            qdrant = QdrantVectorStoreManager._init_online_vector_db(
                url=url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                model=model,
            )
        elif vector_db_mode == "offline":
            qdrant = QdrantVectorStoreManager._init_offline_vector_db(
                vector_store_path=vector_store_path, collection_name=collection_name, model=model
            )
        else:
            raise ValueError("Invalid vector_db_mode. Please provide either 'online' or 'offline'.")
        if qdrant is None:
            raise ValueError("Qdrant client is not initialized.")

        # read the csv file
        df = pd.read_csv(file_path)
        # check that content column exists and url column exists
        if content_column not in df.columns:
            raise ValueError(f"Content column {content_column} not found in the csv file.")
        if url_column not in df.columns:
            raise ValueError(f"URL column {url_column} not found in the csv file.")

        documents = [
            Document(
                page_content=row[content_column],
                metadata={
                    "title": row.get(title_column, ""),
                    "url": row[url_column],
                    "description": row.get(desc_column, ""),
                },
            )
            for row in df.to_dict(orient="records")
        ]

        # split the documents
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=[
                "\n\n",
                "\n",
                ".",
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ",",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                " ",
                "\u200B",  # Zero-width space
                "",
            ],
        )
        split_documents = text_splitter.split_documents(documents)

        # update and save the vector store
        num_batches = (len(split_documents) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(split_documents))
            qdrant.add_documents(
                documents=split_documents[start_idx:end_idx],
                batch_size=batch_size,
            )

        # close the qdrant client
        qdrant.client.close()


import re
import os
from typing import List, Dict, Any


class ArticleTextProcessing:
    @staticmethod
    def preprocess_content(content: str) -> str:
        def clean_paragraph(paragraph: str) -> str:
            lines = paragraph.splitlines()

            # Check if the paragraph is a table
            if any("|" in line for line in lines):
                # If it's a table, return it unchanged
                return "\n".join(lines)

            # For non-table paragraphs, proceed with cleaning
            lines = list(dict.fromkeys(lines))  # Remove duplicates
            lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines and trim
            lines = [re.sub(r"^\s*[^\w\s]+\s*$", "", line) for line in lines]  # Remove orphaned punctuation

            # Remove common code artifacts and notes, but keep markdown headings
            artifact_patterns = [
                r"^\s*//.*$",  # Single-line comments
                r"^\s*/\*.*?\*/\s*$",  # Multi-line comments
                r"^\s*TODO:.*$",  # TODO notes
                r"^\s*FIXME:.*$",  # FIXME notes
                r"^\s*DEBUG:.*$",  # DEBUG notes
            ]
            for pattern in artifact_patterns:
                lines = [line for line in lines if not re.match(pattern, line)]

            return " ".join(lines)  # Join non-table lines

        paragraphs = re.split(r"\n{2,}", content)
        cleaned_paragraphs = [clean_paragraph(p) for p in paragraphs if p.strip()]
        return cleaned_paragraphs

    @staticmethod
    def parse_article_into_dict(input_string: str) -> Dict[str, Any]:
        cleaned_input = ArticleTextProcessing.preprocess_content(input_string)
        # reduce newlines to a maximum of 2
        # cleaned_input = re.sub(r"\n{3,}", "\n\n", cleaned_input)
        # lines = cleaned_input.split("\n")
        root = {"content": "", "subsections": {}}
        current_path = [(root, -1)]  # (current_dict, level)

        for line in cleaned_input:
            if line.startswith("#"):
                level = line.count("#")
                title = line.strip("# ").strip()
                new_section = {"content": "", "subsections": {}}

                while current_path and current_path[-1][1] >= level:
                    current_path.pop()

                current_path[-1][0]["subsections"][title] = new_section
                current_path.append((new_section, level))
            else:
                current_path[-1][0]["content"] += line + "\n\n"

        return root["subsections"]

    @staticmethod
    def parse_markdown_file(file_path: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return ArticleTextProcessing.parse_article_into_dict(content)

    @staticmethod
    def combine_markdown_contents(markdown_files: List[str]) -> Dict[str, Dict[str, Any]]:
        combined_content = {}
        for counter, file in enumerate(markdown_files, 1):
            content = ArticleTextProcessing.parse_markdown_file(file)
            if content:
                file_name = os.path.basename(file)
                prefixed_content = {
                    f"{counter}. {file_name} - {key}": value for key, value in content.items()
                }
                combined_content.update(prefixed_content)
            else:
                print(f"Warning: No content found in file {file}")
        return combined_content

    @staticmethod
    def limit_word_count_preserve_newline(input_string, max_word_count):
        """
        Limit the word count of an input string to a specified maximum, while preserving the integrity of complete lines.

        The function truncates the input string at the nearest word that does not exceed the maximum word count,
        ensuring that no partial lines are included in the output. Words are defined as text separated by spaces,
        and lines are defined as text separated by newline characters.

        Args:
            input_string (str): The string to be truncated. This string may contain multiple lines.
            max_word_count (int): The maximum number of words allowed in the truncated string.

        Returns:
            str: The truncated string with word count limited to `max_word_count`, preserving complete lines.
        """

        word_count = 0
        limited_string = ""

        for word in input_string.split("\n"):
            line_words = word.split()
            for lw in line_words:
                if word_count < max_word_count:
                    limited_string += lw + " "
                    word_count += 1
                else:
                    break
            if word_count >= max_word_count:
                break
            limited_string = limited_string.strip() + "\n"

        return limited_string.strip()

    @staticmethod
    def remove_citations(s):
        """
        Removes all citations from a given string. Citations are assumed to be in the format
        of numbers enclosed in square brackets, such as [1], [2], or [1, 2], etc. This function searches
        for all occurrences of such patterns and removes them, returning the cleaned string.

        Args:
            s (str): The string from which citations are to be removed.

        Returns:
            str: The string with all citation patterns removed.
        """

        return re.sub(r"\[\d+(?:,\s*\d+)*\]", "", s)

    @staticmethod
    def parse_citation_indices(s):
        """
        Extracts citation indexes from the provided content string and returns them as a list of integers.

        Args:
            content (str): The content string containing citations in the format [number].

        Returns:
            List[int]: A list of unique citation indexes extracted from the content, in the order they appear.
        """
        matches = re.findall(r"\[\d+\]", s)
        return [int(index[1:-1]) for index in matches]

    @staticmethod
    def remove_uncompleted_sentences_with_citations(text):
        """
        Removes uncompleted sentences and standalone citations from the input text. Sentences are identified
        by their ending punctuation (.!?), optionally followed by a citation in square brackets (e.g., "[1]").
        Grouped citations (e.g., "[1, 2]") are split into individual ones (e.g., "[1] [2]"). Only text up to
        and including the last complete sentence and its citation is retained.

        Args:
            text (str): The input text from which uncompleted sentences and their citations are to be removed.

        Returns:
            str: The processed string with uncompleted sentences and standalone citations removed, leaving only
            complete sentences and their associated citations if present.
        """

        # Convert citations like [1, 2, 3] to [1][2][3].
        def replace_with_individual_brackets(match):
            numbers = match.group(1).split(", ")
            return " ".join(f"[{n}]" for n in numbers)

        # Deduplicate and sort individual groups of citations.
        def deduplicate_group(match):
            citations = match.group(0)
            unique_citations = list(set(re.findall(r"\[\d+\]", citations)))
            sorted_citations = sorted(unique_citations, key=lambda x: int(x.strip("[]")))
            # Return the sorted unique citations as a string
            return "".join(sorted_citations)

        text = re.sub(r"\[([0-9, ]+)\]", replace_with_individual_brackets, text)
        text = re.sub(r"(\[\d+\])+", deduplicate_group, text)

        # Regex pattern to match sentence endings, including optional citation markers.
        eos_pattern = r"([.!?])\s*(\[\d+\])?\s*"
        matches = list(re.finditer(eos_pattern, text))
        if matches:
            last_match = matches[-1]
            text = text[: last_match.end()].strip()

        return text

    @staticmethod
    def clean_up_citation(conv):
        for turn in conv.dlg_history:
            turn.agent_utterance = turn.agent_utterance[: turn.agent_utterance.find("References:")]
            turn.agent_utterance = turn.agent_utterance[: turn.agent_utterance.find("Sources:")]
            turn.agent_utterance = turn.agent_utterance.replace("Answer:", "").strip()
            try:
                max_ref_num = max([int(x) for x in re.findall(r"\[(\d+)\]", turn.agent_utterance)])
            except Exception as e:
                max_ref_num = 0
            if max_ref_num > len(turn.search_results):
                for i in range(len(turn.search_results), max_ref_num + 1):
                    turn.agent_utterance = turn.agent_utterance.replace(f"[{i}]", "")
            turn.agent_utterance = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                turn.agent_utterance
            )

        return conv

    @staticmethod
    def clean_up_outline(outline, topic=""):
        output_lines = []
        current_level = 0  # To track the current section level

        for line in outline.split("\n"):
            stripped_line = line.strip()

            if topic != "" and f"# {topic.lower()}" in stripped_line.lower():
                output_lines = []

            # Check if the line is a section header
            if stripped_line.startswith("#"):
                current_level = stripped_line.count("#")
                output_lines.append(stripped_line)
            # Check if the line is a bullet point
            elif stripped_line.startswith("-"):
                subsection_header = "#" * (current_level + 1) + " " + stripped_line[1:].strip()
                output_lines.append(subsection_header)

        outline = "\n".join(output_lines)

        # Remove references.
        outline = re.sub(r"#[#]? See also.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? See Also.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Notes.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? References.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? External links.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? External Links.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Bibliography.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Further reading*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Further Reading*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Summary.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Appendices.*?(?=##|$)", "", outline, flags=re.DOTALL)
        outline = re.sub(r"#[#]? Appendix.*?(?=##|$)", "", outline, flags=re.DOTALL)

        return outline

    @staticmethod
    def clean_up_section(text):
        """Clean up a section:
        1. Remove uncompleted sentences (usually due to output token limitation).
        2. Deduplicate individual groups of citations.
        3. Remove unnecessary summary."""

        paragraphs = text.split("\n")
        output_paragraphs = []
        summary_sec_flag = False
        for p in paragraphs:
            p = p.strip()
            if len(p) == 0:
                continue
            if not p.startswith("#"):
                p = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(p)
            if summary_sec_flag:
                if p.startswith("#"):
                    summary_sec_flag = False
                else:
                    continue
            if p.startswith("Overall") or p.startswith("In summary") or p.startswith("In conclusion"):
                continue
            if "# Summary" in p or "# Conclusion" in p:
                summary_sec_flag = True
                continue
            output_paragraphs.append(p)

        return "\n\n".join(output_paragraphs)  # Join with '\n\n' for markdown format.

    @staticmethod
    def update_citation_index(s, citation_map):
        """Update citation index in the string based on the citation map."""
        for original_citation in citation_map:
            s = s.replace(f"[{original_citation}]", f"__PLACEHOLDER_{original_citation}__")
        for original_citation, unify_citation in citation_map.items():
            s = s.replace(f"__PLACEHOLDER_{original_citation}__", f"[{unify_citation}]")

        return s

    @staticmethod
    def combine_markdown_contents(markdown_files: List[str]) -> Dict[str, Dict[str, Any]]:
        combined_content = {}
        counter = 1
        for file in markdown_files:
            content = ArticleTextProcessing.parse_markdown_file(file)
            prefixed_content = {f"{counter}. {key}": value for key, value in content.items()}
            combined_content.update(prefixed_content)
            counter += 1
        return combined_content


class FileIOHelper:
    @staticmethod
    def dump_json(obj, file_name, encoding="utf-8"):
        with open(file_name, "w", encoding=encoding) as fw:
            json.dump(obj, fw, default=FileIOHelper.handle_non_serializable)

    @staticmethod
    def handle_non_serializable(obj):
        return "non-serializable contents"  # mark the non-serializable part

    @staticmethod
    def load_json(file_name, encoding="utf-8"):
        with open(file_name, "r", encoding=encoding) as fr:
            return json.load(fr)

    @staticmethod
    def write_str(s, path):
        with open(path, "w") as f:
            f.write(s)

    @staticmethod
    def load_str(path):
        with open(path, "r") as f:
            return "\n".join(f.readlines())

    @staticmethod
    def dump_pickle(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_markdown_files(folder_path: str) -> List[str]:
        markdown_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".md"):
                    markdown_files.append(os.path.join(root, file))
        return sorted(markdown_files)


class WebPageHelper:
    """Helper class to process web pages.

    Acknowledgement: Part of the code is adapted from https://github.com/stanford-oval/WikiChat project.
    """

    def __init__(self, min_char_count: int = 150, snippet_chunk_size: int = 1000, max_thread_num: int = 10):
        """
        Args:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            max_thread_num: Maximum number of threads to use for concurrent requests (e.g., downloading webpages).
        """
        self.httpx_client = httpx.Client(verify=False)
        self.min_char_count = min_char_count
        self.max_thread_num = max_thread_num
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                ".",
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                ",",
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                " ",
                "\u200B",  # Zero-width space
                "",
            ],
        )

    def download_webpage(self, url: str):
        try:
            res = self.httpx_client.get(url, timeout=4)
            if res.status_code >= 400:
                res.raise_for_status()
            return res.content
        except httpx.HTTPError as exc:
            print(f"Error while requesting {exc.request.url!r} - {exc!r}")
            return None

    def urls_to_articles(self, urls: List[str]) -> Dict:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            htmls = list(executor.map(self.download_webpage, urls))

        articles = {}

        for h, u in zip(htmls, urls):
            if h is None:
                continue
            article_text = extract(
                h,
                include_tables=False,
                include_comments=False,
                output_format="txt",
            )
            if article_text is not None and len(article_text) > self.min_char_count:
                articles[u] = {"text": article_text}

        return articles

    def urls_to_snippets(self, urls: List[str]) -> Dict:
        articles = self.urls_to_articles(urls)
        for u in articles:
            articles[u]["snippets"] = self.text_splitter.split_text(articles[u]["text"])

        return articles
