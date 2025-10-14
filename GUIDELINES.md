## Annotation Guidlines
For each attribute, you label the attribute and a binary value {0,1} to say if this attribute can be extracted from the paper or not
1. **Name** The name of the dataset. We prefer short names like masader, cidar, calliar, etc.
2. **Subsets** It lists all the dialect subsets in the dataset. 
3. **Link** direct link to the dataset. If the dataset is hosted in HuggingFace, we can use the same link in both the Link and HF Link fields. If there is a link from GitHub and a link from HuggingFace, include the link from GitHub. 
4. **HF Link** The HuggingFace link of the dataset.
5. **License** If not in the paper, check the link of the repository. Use unknown if the license is not specified.
6. **Year** the year the paper was published.
7. **Language** this attribute highlights if the dataset is multilingual or not.
8. **Dialect** The main dialect in the dataset. Choose mixed if there are multiple dialects.
9. **Domain** the source or the content  of the data; for example, Domain=Wikipedia means the dataset is extracted from Wikipedia and news articles, if the dataset is extracted from news outlets.
10. **Form** can be text, audio, images, or videos.
11. **Collection Style** How the dataset is extracted, and the annotation strategy of the dataset.
12. **Description** a short description of the dataset. 
13. **Volume** The total samples for the dataset. If the dataset is multilingual, we use the total samples for the Arabic subset.
14. **Unit** We use sentences if the dataset has short samples, even if there are multiple sentences. We use documents for datasets that have long documents, like language modelling, topic classification, etc.
15. **Ethical Risks** "Low" "most likely no ethical risks associated with this dataset", "Medium" "social media datasets or web-extracted datasets", "High" "hate/offensive datasets from social media, or web pages".
16. **Provider** The entity that created the dataset. If there are many affiliations, we can guess from the link or the funds/acknowledgments. Otherwise, list all affiliations. 
17. **Derived From** lists all the datasets that were used to create or derive the current dataset.
18. **Paper Title** The title of the paper.
19. **Paper Link** we use the direct link of the paper, for example, https//arxiv.org/pdf/2504.21677
20. **Script** Script of the dataset. If there are transliteration or Arabizi, use Latin. 
21. **Tokenized** Is the dataset lemmatized or stemmed?
22. **Host** The main repoistory that hosts the dataset.  
23. **Access** Free if the dataset is public. Upon Request, if the dataset is behind a form. With-Fee if the dataset is paid. 
24. **Cost** the cost of the dataset if paid. 
25. **Test Split** True only if the dataset has (training and test splits). If the dataset has only one split, even if it is for testing, we set that to false
26. **Tasks** the list of tasks that this dataset is intended for. Use other if the task doesn’t exist in the options.
27. **Venue Title** the title of the venue i.e. EMNLP, ACL, etc.
28. **Venue Type** the veneue type.
29. **Venue Name** the full name of the venue. for arXiv we don't use the full name.
30. **Authors** the authors of the paper in an ordered fashion.
31. **Affiliations** list only the affiliations without repetition
32. **Abstract** the full abstract of the paper. 