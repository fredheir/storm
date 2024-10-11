import copy
from typing import Union, List, Dict

import dspy

from .storm_dataclass import StormArticle, ArticleSectionNode
from ...interface import ArticlePolishingModule
from ...utils import ArticleTextProcessing


class StormArticlePolishingModule(ArticlePolishingModule):
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage.
    """

    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm
        self.polish_page = PolishPageModule(
            article_gen_lm=self.article_gen_lm, article_polish_lm=self.article_polish_lm
        )

    def polish_article(
        self, topic: str, draft_article: StormArticle, remove_duplicate: bool = False
    ) -> StormArticle:
        article_text = draft_article.to_string()
        polished_article = self.polish_page(
            topic=topic,
            draft_article=draft_article,
            article_text=article_text,
            remove_duplicate=remove_duplicate,
        )
        return polished_article


class IdentifySectionsToPolish(dspy.Signature):
    """You are a faithful text editor that is good at finding repeated information in the article and deleting them to make sure there is no repetition in the article. You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Identify sections of the article that need polishing."""

    draft_page = dspy.InputField(desc="The full text of the draft article")

    sections_to_polish = dspy.OutputField(
        desc="List of sections that need polishing, including their path and content"
    )


class PolishIdentifiedSections(dspy.Signature):
    """You are a faithful text editor that is good at finding repeated information in the article and deleting them to make sure there is no repetition in the article. You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Do your job for the list of sections of the article identified as needing polishing."""

    sections_to_polish = dspy.InputField(
        desc="List of sections that need polishing, including their path and content"
    )

    polished_sections = dspy.OutputField(
        desc="Polished sections in a structured markdown format, maintaining the original hierarchy. Each section should start with the appropriate number of '#' characters to indicate its level in the hierarchy, followed by the section title. The section content should follow on the next line."
    )


class PolishPageModule(dspy.Module):
    def __init__(
        self,
        article_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        article_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm
        self.identify_sections = dspy.Predict(IdentifySectionsToPolish)
        self.polish_sections = dspy.Predict(PolishIdentifiedSections)

    def forward(
        self, topic: str, draft_article: StormArticle, article_text: str, remove_duplicate: bool = False
    ):

        with dspy.settings.context(lm=self.article_polish_lm):
            sections_to_polish = self.identify_sections(
                draft_page=article_text,
            ).sections_to_polish

            polished_sections = self.polish_sections(
                sections_to_polish=sections_to_polish,
            ).polished_sections

        # Merge polished sections back into the original StormArticle
        polished_article = self.merge_polished_sections(draft_article, polished_sections)
        return polished_article

    def merge_polished_sections(self, draft_article: StormArticle, polished_sections: str) -> StormArticle:
        # Parse the polished sections into a dictionary structure
        polished_dict = ArticleTextProcessing.parse_article_into_dict(polished_sections)

        # Use the existing method to merge the polished content into the draft article
        draft_article.insert_or_create_section(article_dict=polished_dict, trim_children=True)

        # Perform post-processing
        draft_article.post_processing()

        return draft_article


class WriteLeadSection(dspy.Signature):
    """Write an executive summary for the given article with the following guidelines:
    1. The executive summary should stand on its own as a concise overview of the article's main findings. It should identify the topic, establish context, explain why the topic is notable, and summarize the most important points, including any prominent controversies.
    2. The executive summary should be concise and contain no more than four well-composed paragraphs.
    3. The executive summary should be carefully sourced as appropriate. Add inline citations (e.g., "Washington, D.C., is the capital of the United States.[1][3].") where necessary.
    """

    topic = dspy.InputField(prefix="The topic of the page: ", format=str)
    draft_page = dspy.InputField(prefix="The draft page:\n", format=str)
    lead_section = dspy.OutputField(prefix="Write the lead section:\n", format=str)


class PolishPage(dspy.Signature):
    """You are a faithful text editor that is good at finding repeated information in the article and deleting them to make sure there is no repetition in the article. You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Do your job for the following article."""

    draft_page = dspy.InputField(prefix="The draft article:\n", format=str)
    page = dspy.OutputField(prefix="Your revised article:\n", format=str)


class EnglishPage(dspy.Signature):
    """You are a faithful text editor. Your task is to ensure that the text is written in British English. You won't delete any material. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Do your job for the following article."""

    draft_page = dspy.InputField(prefix="The draft article:\n", format=str)
    page = dspy.OutputField(prefix="Your revised article:\n", format=str)
