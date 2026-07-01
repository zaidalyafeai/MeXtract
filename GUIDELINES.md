## Annotation Guidlines
For each attribute, you label the attribute and a binary value {0,1} to say if this attribute can be extracted from the paper or not
1. **Name** The name of the dataset. We prefer short names like masader, cidar, calliar, etc.
2. **Dialect Subsets** It lists all the dialect subsets in the dataset.
3. **HF Link** The HuggingFace link of the dataset.
4. **Link** direct link to the dataset. If the dataset is hosted in HuggingFace, we can use the same link in both the Link and HF Link fields. If there is a link from GitHub and a link from HuggingFace, include the link from GitHub.
5. **License** If not in the paper, check the link of the repository. Use unknown if the license is not specified.
6. **Year** the year the paper was published.
7. **Language** this attribute highlights if the dataset is multilingual or not.
8. **Dialect** The main dialect in the dataset. Choose mixed if there are multiple dialects.
9. **Source** the source or the content of the data; for example, Source=Wikipedia means the dataset is extracted from Wikipedia and news articles, if the dataset is extracted from news outlets.
10. **Domain** the domain of the content; for example, religion, health, news, education, science, literature, etc.
11. **Form** can be text, audio, images, or videos.
12. **Annotation Style** How the dataset is extracted, and the annotation strategy of the dataset; for example, human annotation, machine annotation, LLM annotation, metadata-derived, inherited annotation.
13. **Description** a short description of the dataset.
14. **Volume** The total samples for the dataset. If the dataset is multilingual, we use the total samples for the Arabic subset.
15. **Unit** We use sentences if the dataset has short samples, even if there are multiple sentences. We use documents for datasets that have long documents, like language modelling, topic classification, etc.
16. **Ethical Risks** "Low" "most likely no ethical risks associated with this dataset", "Medium" "social media datasets or web-extracted datasets", "High" "hate/offensive datasets from social media, or web pages".
17. **Provider** The entity that created the dataset. If there are many affiliations, we can guess from the link or the funds/acknowledgments. Otherwise, list all affiliations.
18. **Derived From** lists all the datasets that were used to create or derive the current dataset.
19. **Paper Title** The title of the paper.
20. **Paper Link** we use the direct link of the paper, for example, https//arxiv.org/pdf/2504.21677
21. **Script** Script of the dataset. If there are transliteration or Arabizi, use Latin.
22. **Tokenized** Is the dataset lemmatized or stemmed?
23. **Host** The main repoistory that hosts the dataset.
24. **Access** Free if the dataset is public. Upon Request, if the dataset is behind a form. With-Fee if the dataset is paid.
25. **Cost** the cost of the dataset if paid.
26. **Has Splits** True only if the dataset has (training and test splits). If the dataset has only one split, even if it is for testing, we set that to false.
27. **Partial** True if the dataset is a partial release or a subset of a larger dataset.
28. **Tasks** the list of tasks that this dataset is intended for. Use other if the task doesn’t exist in the options.
29. **Venue Title** the title of the venue i.e. EMNLP, ACL, etc.
30. **Venue Type** the veneue type.
31. **Venue Name** the full name of the venue. for arXiv we don't use the full name.
32. **Authors** the authors of the paper in an ordered fashion.
33. **Affiliations** list only the affiliations without repetition
34. **Abstract** the full abstract of the paper.
