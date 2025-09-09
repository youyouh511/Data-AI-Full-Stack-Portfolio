
# Youyou Huang | Data & AI & Full-Stack Portfolio

*Welcome to Youyou's portfolio.* 

This collection features selected projects that reflect my technical expertise and growth. While many were completed in collaborative or proprietary environments and cannot be publicly shared, each represents a meaningful milestone in my development across data analytics, software development, and product-oriented problem solving.


## Completed
### Operational Design Domain Safety Visualization Tool
**Domain:** Software Development, Geospatial Analytics, Autonomous Systems

**Client Partner:** [TierIV](https://tier4.jp/en/)

**Course:** Capstone Project, Carnegie Mellon University

**Timeline:** Summer 2025

**Tech Stack:** Python, Flask, MongoDB, React, OSMnx, GeoPandas, NetworkX, Graph Database

**Demo:** *Contact for discussion (code and report not publicly available)*

We built a modular visualization tool to map Operational Design Domains (ODDs) for autonomous vehicle deployment, enabling safety-focused analysis of road networks across geographic regions. As technical lead, I architected the full-stack pipeline and contributed across backend and frontend development.

- Designed the end-to-end application pipeline, including IO schema, caching strategy, and modular backend architecture for scalable geospatial analysis.
- Implemented core backend logic for feature extraction and network filtering, integrating OSMnx and MongoDB to support high-performance querying and storage.
- Contributed significantly to frontend delivery, including dynamic feature filtering.
- Led system design discussions and coordinated cross-functional development across data processing, visualization, and strategic framing.
- Co-authored final report and presentation materials.


### [STAR+: Training-Free Product Recommendations with LLM](https://github.com/Bernie-cc/Training-Free-Recommendations-with-LLM/tree/main)
**Domain:** NLP, LLM, Recommender Systems

**Course:** 11-711 Advanced Natural Language Processing, Carnegie Mellon University

**Timeline:** Spring 2025

**Tech Stack**: Python, OpenAI API, SentenceTransformers, CLIP, ViT, Collaborative Filtering, Promp Engineering

**Report:** [Request access](https://drive.google.com/file/d/1ruaWch6242IuS6Oz5qAXAxJtCeciaRW-/view?usp=drive_link)

We explored training-free product recommendation using dense retrieval, collaborative filtering, and large language models (LLMs). While not the primary coder, I played a key role in shaping the research direction, experimental design, and final deliverables.

- Led the literature review and co-developed the methodological framework, grounding the system in recent advances in zero-shot and retrieval-based recommendation.
- Designed and tested an experimental pipeline integrating CLIP/ViT and SentenceTransformers to assess multimodal alignment strategies.
- Conducted ablation studies and collaborated on testing and evaluation to analyze the effects of embedding configurations, scoring parameters, and prompt formats on recommendation quality.
- Co-authored the final report and led the poster presentation, translating technical insights into accessible narratives for diverse audiences.


### [Causal Graph Neural Networks for Wildfire Prediction](https://github.com/youyouh511/11785_IDL_S25_Final-Project)

**Domain:** Deep Learning, Spatiotemporal Modeling, Causal Inference, Graph Neural Network

**Course:** 11-785 Introduction to Deep Learning, Carnegie Mellon University

**Timeline:** Spring 2025

**Tech Stack:** Python, PyTorch, LSTM, DenseGCNConv, PCMCI (Tigramite)

**Report:** [Request access](https://drive.google.com/file/d/1A_GaWo9ynqKYUfVtrftKub3_iiRmtQ97/view?usp=sharing)

We reimplemented and extended a causal graph neural network framework for wildfire danger prediction using the SeasFire datacube. As technical lead, I drove the research direction, model architecture, sampling strategy, and final report synthesis.

- Led literature review and methodological framework development.
- Proposed and implemented a granular sampling strategy to isolate fire emergence after extended calm periods, refining the research focus and improving label quality.
- Designed and coded the full model pipeline, integrating PCMCI-generated causal graphs with LSTM and DenseGCNConv layers for temporal-spatial prediction.
- Conducted performance analysis and ablation studies to assess model robustness.
- Co-authored the final report, synthesizing technical insights, evaluation outcomes, and future directions for scalable, interpretable wildfire forecasting.


### [Retrieval-Augmented Generation (RAG) Question Answering System](https://github.com/Bernie-cc/RAG-based-Question-Answering-System)
**Domain:** Natural Language Processing (NLP), Large Language Model(LLM), Retrieval-Augmented Generation (RAG)

**Course:** 11-711 Advanced Natural Language Processing, Carnegie Mellon University

**Timeline:** Spring 2025

**Tech Stack:** Python, LangChain, HuggingFace, SentenceTransformers, ChromaDB, Web Scraping (BeautifulSoup/Selenium), Prompt Engineering

**Report:** [Request access](https://drive.google.com/file/d/1Ihw45AsyRrwwqxNuHV_w35zz4K09SEYE/view?usp=drive_link)

We built a RAG-based QA system that integrates document retrieval with generative models to answer questions grounded in a custom knowledge base. While not the primary pipeline developer, I played a key role in research, experimentation, and delivery.

- Researched and evaluated optimal models for system components (baseline LLMs, dense retrievers, and re-rankers) to balance performance and efficiency.
- Built the knowledge corpus via targeted web scraping of CMU- and Pittsburgh-related sources.
- Validated LLM-annotated documents to support training and evaluation workflows.
- Conducted ablation studies across prompt formats and fine-tuned retriever parameters to improve relevance search.
- Led the drafting and delivery of the final system report, synthesizing technical insights and evaluation outcomes.


### [Judgment by Algorithm: Exploring AI Fairness in Criminal Justice](https://github.com/youyouh511/RAI/tree/main)

**Domain:** Responsible AI

**Course:** 94-885 Responsible AI - Fundamentals, Policy, and Implementation

**Timeline:** Fall 2024

**Tech Stack:** Python, Pandas, Numpy, Data Analytics, Data Visualization

We investigated algorithmic bias in the criminal justice system using the COMPAS Recidivism Risk Score dataset, uncovering disparities in fairness metrics across demographic groups.

- Served as the primary lead for statistical analysis and data visualization, driving the project's empirical backbone.
- Contributed extensively to the final report and peer review process, ensuring clarity and alignment with Responsible AI principles.



## In Progress
### End-Goal-Oriented Employment Capacity Mapping & Building App
**Independent Project**

**Domain:** App Design, Software Development, Project Management, Natural Language Processing (NLP), Large Language Models (LLM)

**Timeline:** 2025

**Tech Stack:** Python, FastAPI, [Java], MongoDB, HuggingFace, SentenceTransformers, Web Scraping (BeautifulSoup/Selenium)

This ongoing independent project aims to streamline career planning by aligning individual qualifications with job market demands. The app ingests job postings (via web scraping or user uploads), resumes, and course syllabi (completed or planned), then maps out matching skillsets using keyword matching, dense retrieval, and LLM-based synthesis.

- Designing and implementing the full front and backend architecture, including IO schema, data ingestion pipelines, and modular APIs for resume and syllabus parsing.
- Developing matching algorithms that combine keyword heuristics with semantic similarity via dense embeddings and transformer-based models.
- Integrating a granular skill-mapping strategy to highlight gaps and strengths across academic, experiential, and aspirational dimensions.
- Building support modules for resume refinement, project tagging, and skill reorganization based on target job criteria.
- Designing a dynamic "To-Do" generator that outputs personalized courses of action to take, projects to build, and skills to acquire to meet targeted job requirements.
