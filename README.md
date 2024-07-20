# Document

## Introduction

While Jupyter Notebooks are excellent for data exploration and interactive development, their structure 
presents significant challenges when used for production-level software development. During the interview 
for the GenAI Senior Data Scientist role, it became evident that producing robust, maintainable code 
suitable for production environments is crucial. Consequently, I opted to use PyCharm for its comprehensive 
support in professional software development. The decision was driven by several limitations inherent to 
Jupyter Notebooks when deploying software in a production setting, including:

- **Lack of Modularity**: Jupyter Notebooks' linear, cell-based structure hampers code reuse and modularity in larger systems, challenging the integration into modular, scalable production architectures. Source: Ploomber.io

- **Version Control Challenges**: Notebooks mix code, output, and metadata in a JSON format, complicating version control practices. The `.ipynb` file format results in illegible `git diff` outputs, making code reviews difficult. Additionally, if notebooks include images, the file size can increase significantly, potentially bloating git repositories. Source: Nepture.ai

- **Poor Error Handling**: The interactive, stateful nature of notebooks can lead to hidden states that persist unpredictably, complicating debugging and obscuring errors until runtime. Source: Ploomber.io

- **Concurrency Limitations**: Designed for single-user, interactive sessions, Jupyter Notebooks do not naturally support high concurrency or integration into distributed systems, limiting their effectiveness in scalable, real-time production settings. Source: towardsdatascience.com

- **Dependency Management**: Jupyter Notebooks often lead to challenges in managing dependencies and ensuring consistent environments across different stages of development, contributing to "it works on my machine" issues when moving to production. Source: Neptune.ai

## Question 1

To enhance the Mean Average Precision at 10 (MAP@10) of the search engine, several strategic updates are 
proposed. First, enhanced text preprocessing techniques such as advanced tokenization will capture 
product-specific nuances, such as hyphenations and compound words, ensuring all relevant text data is 
utilized effectively. This prevents the loss of crucial contextual information in product descriptions, 
thereby improving the accuracy of matching user queries. Additionally, utilizing a domain-specific stopword 
list will minimize noise in the textual data, allowing the algorithm to focus on meaningful words, thus likely 
enhancing the relevance of search results.

Further, tuning the parameters of the TF-IDF vectorization, specifically adjusting `max_df` and `min_df`, 
helps exclude terms that are either too common or too rare. This refinement focuses the search on terms that 
best differentiate product relevance, creating a more effective search space and likely improving result 
relevance. Implementing sublinear term frequency scaling further balances term weights based on their 
informational value rather than frequency, which can highlight key terms that might otherwise be overlooked.

Semantic enhancement through the incorporation of pre-trained word embeddings allows the search engine to 
understand deeper linguistic relationships and contextual similarities between terms. This capability enables 
the engine to recognize relevant products based on semantic context, not just exact word matches, 
significantly broadening the scope of search queries to fetch more pertinent results.

Query expansion with synonyms further augments this approach by broadening the search query to include 
synonyms and related terms, ensuring that variations in user language do not cause relevant products to be 
overlooked. This synonym expansion facilitates a better alignment of search results with user expectations by accommodating different phrasings of similar concepts. Additionally, the incorporation of contextual understanding through synonyms enhances the search engine’s comprehension of user intent, which is crucial for matching the nuanced demands of diverse queries.

Lastly, implementing a feedback loop that leverages click-through data from real user interactions offers dynamic improvements to the search algorithms. By analyzing which results users engage with, the system can continuously refine its processes to prioritize more relevant results. This adaptive learning process is expected to dynamically improve the search engine's accuracy and relevance based on direct user feedback, contributing to an improved MAP@10 score by ensuring that the results more closely align with user needs and preferences.

## Question 2

To address the current limitation where partial matches in the search engine are treated as irrelevant, a refined evaluation function, `map_at_k`, has been developed. This function incorporates a mechanism to fairly score partial matches by assigning them a fractional weight during the MAP@K computation. Specifically, the function includes an optional `partial_matches` parameter that, when enabled, adjusts the scoring to allocate a fractional score of 0.5 to partial matches, in contrast to a full score for exact matches. This method not only recognizes the relevance of partial matches—which may not be perfect but still hold significant value—but also enhances the overall assessment of the search performance by acknowledging the varying degrees of match accuracy.

The modified function for evaluating the search engine's performance introduces a more nuanced scoring method by integrating a weighted scoring for partial matches. This adjustment effectively addresses the previous model's overly strict penalization, which failed to acknowledge the value of partially relevant results. By assigning a fractional weight of 0.5 to partial matches, the function strikes a balance that recognizes their significance without equating them to exact matches. Additionally, the inclusion of the `partial_matches` flag enhances the flexibility of the scoring system. This allows the evaluation process to be adapted according to different relevance criteria, supporting varying levels of strictness based on specific needs or contexts, thereby ensuring a more tailored and effective assessment.

### Pros and Cons

**Pros**

- **Enhanced Fairness**: Incorporating partial matches with a fractional score reduces the penalty for non-exact matches, leading to a more balanced and fair evaluation of search results.
  
- **Increased Sensitivity**: The function improves the sensitivity of performance metrics, capturing a broader spectrum of useful search results, which may have been previously overlooked.

**Cons**

- **Complexity in Tuning**: Introducing the `partial_matches` parameter requires careful consideration and testing to determine the optimal configuration for balancing the scoring of partial matches, adding a layer of complexity to configuring the search engine.
  
- **Potential for Overestimation**: If not carefully calibrated, the weight assigned to partial matches could lead to an overestimation of the system's effectiveness, especially if many partial matches do not contribute significantly to user satisfaction.

To enhance the evaluation of the search engine further, incorporating additional metrics such as precision, recall, and F1-score at K is recommended. These metrics offer deeper insights into the balance between precision and recall, enriching our understanding of the search engine's operational effectiveness. Precision at K, which measures the proportion of retrieved documents that are relevant, provides direct feedback on the accuracy of the search results. Meanwhile, Recall at K assesses the extent to which the system retrieves all relevant documents within the top K results, a crucial measure for evaluating the comprehensiveness of the search output. Utilizing these metrics in combination enables a robust analysis of the search engine's performance, guiding further optimizations and providing stakeholders with comprehensive insights into various aspects of system efficacy. This holistic approach to evaluation ensures that improvements can be strategically targeted, thereby enhancing both the user experience and the technical robustness of the search functionality.

## Question 3

In the pursuit of transitioning our search engine from a research prototype in Jupyter Notebooks to a production-ready application, a series of systematic enhancements and refactoring were undertaken. The initial step involved converting the core Jupyter Notebook (.ipynb) into a more manageable and version-controllable Python (.py) script. This change facilitated the next logical step: establishing a version control system. A Git repository was created, with the original files committed to the main branch to set a baseline for future modifications.

To segregate development efforts from the stable production version, a dedicated 'dev' branch was created. This environment allowed for iterative development and testing without affecting the main codebase. During the initial code review, redundant imports, such as unused instances of the numpy library, were removed to streamline the code. Additionally, comments were shifted from block annotations above functions to inline comments alongside the code, enhancing readability and maintainability.

The refinement process extended into the core functionality of the search engine. Attempts were made to incorporate both stemming and lemmatization techniques to the text processing pipeline, aiming to improve the precision of the search results. However, these adjustments did not yield the anticipated benefits; stemming negatively impacted the score, and lemmatization showed no significant effect. A pivotal improvement was achieved by modifying the vectorizer settings, which led to a noticeable enhancement in performance metrics. Further experimenting with the inclusion of n-grams slowed down the computation but substantially increased the score, demonstrating a trade-off between efficiency and accuracy.

One experimental modification involved integrating synonyms into the query processing to potentially widen the search scope and capture more relevant results. Surprisingly, this adjustment resulted in a decrease in performance, highlighting the complexity of natural language processing and the need for finely-tuned semantic understanding.

A significant breakthrough was achieved with the introduction of a partial match scoring system within the evaluation metrics. By acknowledging partial matches in the search results, the search engine's score improved dramatically to 77, reflecting a more nuanced assessment of search accuracy.

To bolster the robustness of the application, comprehensive error handling and logging mechanisms were integrated. These enhancements not only aid in monitoring the system's performance but also facilitate quicker debugging and resolution of issues.

Finally, the architecture of the program was overhauled to become more modular. Separate classes were defined for handling data processing, search functionality, and performance metrics evaluation. This modularization was further supported by the creation of a processing pipeline, which not only organized the flow of data through various stages of the search and evaluation but also made the codebase easier to manage and extend.

Each of these steps was guided by the dual objectives of enhancing the search engine’s performance and ensuring its readiness for deployment in a production environment. The iterative improvements and rigorous testing underscore the commitment to delivering a robust, efficient, and effective search solution.
