

import json
import random
import uuid
import pandas as pd

def generate_synthetic_dataset(num_rows=10000):
    data = []
    sentence_templates = {
        "self_mentions": {
            "true": [
                {"template": "In our recent study, {marker} that the observed anomalies necessitate a re-evaluation of existing theoretical frameworks.", "marker": "we argue"},
                {"template": "This paper {marker} a novel approach to understanding cognitive biases in decision-making.", "marker": "demonstrates"},
                {"template": "I {marker} that the preliminary results indicate a significant correlation between variable X and variable Y.", "marker": "believe"},
                {"template": "Our {marker} suggests a new interpretation of the developmental trajectory.", "marker": "analysis"},
                {"template": "We {marker} a framework for assessing the impact of climate change on biodiversity.", "marker": "propose"},
                {"template": "In this investigation, we {marker} the implications of quantum entanglement for information theory.", "marker": "explore"},
                {"template": "The authors {marker} that the findings challenge conventional wisdom regarding economic growth.", "marker": "contend"},
                {"template": "Through our empirical work, we {marker} the efficacy of this intervention.", "marker": "have shown"},
                {"template": "I {marker} that the current model is insufficient to explain the observed phenomena.", "marker": "assert"},
                {"template": "Our {marker} indicates a clear deviation from the expected pattern.", "marker": "research"},
            ],
            "edge": [
                {"template": "We {marker} to the conference last week, and it was a truly insightful experience for our team.", "marker": "went"},
                {"template": "I {marker} about my family when I am away from home.", "marker": "think"},
                {"template": "Our {marker} is parked in the driveway.", "marker": "car"},
                {"template": "My {marker} on the new policy is that it will have a limited impact on daily operations.", "marker": "opinion"},
                {"template": "I {marker} to finish this report by tomorrow morning.", "marker": "need"},
                {"template": "We {marker} a great time at the workshop.", "marker": "had"},
                {"template": "My {marker} is to become a leading expert in this field.", "marker": "goal"},
                {"template": "We {marker} to the beach every summer.", "marker": "go"},
                {"template": "I {marker} that you are doing well.", "marker": "hope"},
                {"template": "Our {marker} is to provide high-quality education.", "marker": "mission"},
            ]
        },
        "hedges": {
            "true": [
                {"template": "The observed discrepancies {marker} a potential limitation in the current experimental design.", "marker": "might suggest"},
                {"template": "The preliminary results {marker} a trend towards increased efficiency under these novel conditions.", "marker": "appear to indicate"},
                {"template": "It {marker} that the interaction effect between the two variables contributes significantly to the outcome.", "marker": "seems probable"},
                {"template": "This finding {marker} that the hypothesis requires further refinement.", "marker": "may imply"},
                {"template": "To {marker}, the data supports the initial premise.", "marker": "some extent"},
                {"template": "It {marker} that the correlation is not purely coincidental.", "marker": "is possible"},
                {"template": "The evidence {marker} a causal relationship, though further studies are needed.", "marker": "could suggest"},
                {"template": "The results {marker} to be consistent with previous literature.", "marker": "seem"},
                {"template": "This {marker} indicates a potential area for future research.", "marker": "tentatively"},
                {"template": "The model {marker} to capture the complexity of the system.", "marker": "appears"},
            ],
            "edge": [
                {"template": "He {marker} to the library later if he finishes his work on time.", "marker": "might go"},
                {"template": "It {marker} like a good idea to implement the new security protocols immediately.", "marker": "seems"},
                {"template": "You {marker} leave now if you wish.", "marker": "may"},
                {"template": "I {marker} help you with that task.", "marker": "could"},
                {"template": "Perhaps {marker} will be better tomorrow.", "marker": "the weather"},
                {"template": "It {marker} that the train is delayed.", "marker": "appears"},
                {"template": "She {marker} be late for the meeting.", "marker": "might"},
                {"template": "We {marker} consider all options before making a decision.", "marker": "should"},
                {"template": "It {marker} to be a beautiful day.", "marker": "seems"},
                {"template": "They {marker} arrive any minute now.", "marker": "could"},
            ]
        },
        "boosters": {
            "true": [
                {"template": "It {marker} that the proposed model offers a superior fit for the empirical data.", "marker": "is clear"},
                {"template": "The experimental evidence {marker} the efficacy of this novel therapeutic intervention.", "marker": "certainly proves"},
                {"template": "This analysis {marker} a significant and robust effect across all tested conditions.", "marker": "undoubtedly demonstrates"},
                {"template": "The findings {marker} that the hypothesis is unequivocally supported.", "marker": "clearly show"},
                {"template": "There {marker} that this methodology is highly effective.", "marker": "is no doubt"},
                {"template": "The data {marker} the validity of the theoretical framework.", "marker": "conclusively confirms"},
                {"template": "It is {marker} that the results are consistent with the predictions.", "marker": "evident"},
                {"template": "The study {marker} the critical role of this variable.", "marker": "definitively establishes"},
                {"template": "This {marker} indicates a strong correlation.", "marker": "unquestionably"},
                {"template": "The evidence {marker} the proposed mechanism.", "marker": "strongly supports"},
            ],
            "edge": [
                {"template": "He was {marker} happy with the results of his exam, which was certainly a relief.", "marker": "very"},
                {"template": "Clearly, {marker} need to review the instructions before attempting the next step.", "marker": "you"},
                {"template": "That was {marker} an amazing performance!", "marker": "absolutely"},
                {"template": "I am {marker} sure that I locked the door.", "marker": "quite"},
                {"template": "The movie was {marker} boring.", "marker": "really"},
                {"template": "She is {marker} talented.", "marker": "extremely"},
                {"template": "It was {marker} a difficult decision.", "marker": "truly"},
                {"template": "He {marker} understood the concept.", "marker": "fully"},
                {"template": "The food was {marker} delicious.", "marker": "incredibly"},
                {"template": "This is {marker} the best solution.", "marker": "definitely"},
            ]
        },
        "frame_markers": {
            "true": [
                {"template": "{marker}, we will delineate the theoretical underpinnings of the proposed framework.", "marker": "Firstly"},
                {"template": "In {marker}, this paper has systematically demonstrated the critical role of metacognitive strategies.", "marker": "conclusion"},
                {"template": "The {marker} discusses the methodological considerations and the data collection procedures.", "marker": "next section"},
                {"template": "To {marker}, the findings highlight the importance of interdisciplinary collaboration.", "marker": "summarize"},
                {"template": "{marker}, we present the implications of our research for educational policy.", "marker": "Finally"},
                {"template": "This {marker} aims to provide a comprehensive overview of recent advancements in AI.", "marker": "paper"},
                {"template": "As {marker}, the previous chapter explored the historical context of this phenomenon.", "marker": "mentioned above"},
                {"template": "In {marker} of this discussion, we emphasize the need for further empirical studies.", "marker": "light"},
                {"template": "The {marker} of this study is to investigate the impact of social media on political discourse.", "marker": "purpose"},
                {"template": "To {marker}, the results indicate a significant shift in consumer behavior.", "marker": "conclude"},
            ],
            "edge": [
                {"template": "{marker}, I woke up, then I had breakfast, and finally, I left for work.", "marker": "First"},
                {"template": "The {marker} time I visited that museum, they had a fascinating exhibit on ancient civilizations.", "marker": "last"},
                {"template": "Finally, {marker} arrived at the destination after a long journey.", "marker": "he"},
                {"template": "In {marker} opinion, the new regulations are too restrictive.", "marker": "my"},
                {"template": "First, {marker} need to gather all the necessary materials.", "marker": "I"},
                {"template": "The {marker} chapter of the book was very exciting.", "marker": "final"},
                {"template": "To {marker} a long story short, we decided to go home.", "marker": "make"},
                {"template": "In {marker} words, it was a disaster.", "marker": "other"},
                {"template": "The {marker} step is to analyze the data.", "marker": "next"},
                {"template": "Finally, {marker} can relax after a busy week.", "marker": "we"},
            ]
        },
        "code_glosses": {
            "true": [
                {"template": "The phenomenon exhibits high plasticity, {marker}, its ability to adapt to varying environmental conditions.", "marker": "that is to say"},
                {"template": "Several factors influence the outcome, {marker}, temperature, pressure, and catalyst concentration.", "marker": "for example"},
                {"template": "The study focused on specific cognitive biases, {marker}, confirmation bias and anchoring bias.", "marker": "namely"},
                {"template": "The methodology involves several stages, {marker}, data collection, analysis, and interpretation.", "marker": "including"},
                {"template": "The concept of 'qualia' {marker} subjective conscious experiences.", "marker": "in other words"},
                {"template": "The experiment used a double-blind design, {marker}, neither the participants nor the researchers knew who received the treatment.", "marker": "i.e."},
                {"template": "Common statistical tests {marker} t-tests, ANOVA, and regression analysis.", "marker": "e.g."},
                {"template": "The process is iterative, {marker}, it involves repeated cycles of refinement.", "marker": "that is"},
                {"template": "The research explores various forms of renewable energy, {marker}, solar, wind, and geothermal.", "marker": "such as"},
                {"template": "The findings have broad implications, {marker}, for public health policy and urban planning.", "marker": "specifically"},
            ],
            "edge": [
                {"template": "{marker} a dog or a cat, many pets require significant care and attention from their owners.", "marker": "Such as"},
                {"template": "{marker} why I decided to pursue a career in scientific research.", "marker": "That is"},
                {"template": "Including {marker}, everyone attended the meeting.", "marker": "me"},
                {"template": "He enjoys various outdoor activities, {marker} hiking and camping.", "marker": "such as"},
                {"template": "That is {marker} I feel about the situation.", "marker": "how"},
                {"template": "I like many fruits, {marker} apples and bananas.", "marker": "including"},
                {"template": "That is {marker} I learned to play the guitar.", "marker": "how"},
                {"template": "He is good at many sports, {marker} basketball and soccer.", "marker": "such as"},
                {"template": "That is {marker} I will be there.", "marker": "when"},
                {"template": "Including {marker}, the whole team celebrated the victory.", "marker": "him"},
            ]
        },
        "engagement_markers": {
            "true": [
                {"template": "{marker} that this methodological approach differs significantly from traditional qualitative analyses.", "marker": "Note"},
                {"template": "{marker} how these findings might influence future policy decisions regarding environmental sustainability.", "marker": "Consider"},
                {"template": "{marker} observe the consistent trend in the data, which points to a robust effect.", "marker": "You should"},
                {"template": "{marker} us examine the implications of these results for practical applications.", "marker": "Let"},
                {"template": "As {marker}, the previous section highlighted the limitations of the existing models.", "marker": "you can see"},
                {"template": "It {marker} to note that the sample size was relatively small.", "marker": "is important"},
                {"template": "{marker} that the statistical significance was achieved despite the small effect size.", "marker": "Observe"},
                {"template": "{marker} the potential ethical considerations before proceeding with the experiment.", "marker": "Reflect on"},
                {"template": "{marker} that the findings are consistent across different cultural contexts.", "marker": "It is worth noting"},
                {"template": "{marker} the nuances of this complex phenomenon.", "marker": "Appreciate"},
            ],
            "edge": [
                {"template": "{marker} a highly motivated student, and your dedication is commendable.", "marker": "You are"},
                {"template": "What {marker} you think about the latest developments in artificial intelligence?", "marker": "do"},
                {"template": "Can {marker} tell me the time?", "marker": "you"},
                {"template": "If {marker} want to succeed, you must work hard.", "marker": "you"},
                {"template": "You {marker} go to the party if you finish your homework.", "marker": "can"},
                {"template": "What {marker} you do yesterday?", "marker": "did"},
                {"template": "You {marker} a great job on that project.", "marker": "did"},
                {"template": "How {marker} you feel today?", "marker": "do"},
                {"template": "You {marker} always welcome here.", "marker": "are"},
                {"template": "Where {marker} you going?", "marker": "are"},
            ]
        },
        "transitions": {
            "true": [
                {"template": "{marker}, previous meta-analyses indicate a different pattern of results, suggesting a need for further investigation.", "marker": "However"},
                {"template": "{marker}, based on the cumulative evidence, we can confidently conclude that the hypothesis is supported.", "marker": "Therefore"},
                {"template": "{marker}, the qualitative data provides rich insights that complement the quantitative findings.", "marker": "Furthermore"},
                {"template": "The first approach focuses on quantitative analysis; {marker}, the second emphasizes qualitative interpretation.", "marker": "in contrast"},
                {"template": "The initial results were promising; {marker}, subsequent experiments yielded inconsistent data.", "marker": "nevertheless"},
                {"template": "The study employed a mixed-methods design; {marker}, the findings offer a holistic understanding of the phenomenon.", "marker": "consequently"},
                {"template": "The two theories share several common assumptions; {marker}, they diverge significantly in their predictive power.", "marker": "similarly"},
                {"template": "The data suggests a strong correlation; {marker}, a causal link cannot be definitively established.", "marker": "thus"},
                {"template": "The model accounts for various confounding factors; {marker}, its predictive accuracy is enhanced.", "marker": "hence"},
                {"template": "The research highlights the importance of early intervention; {marker}, it provides practical recommendations for policymakers.", "marker": "moreover"},
            ],
            "edge": [
                {"template": "{marker} I went home, and after that, I prepared dinner for my family.", "marker": "Then"},
                {"template": "First, {marker} need to gather all the necessary materials before starting the experiment.", "marker": "I"},
                {"template": "After {marker}, I went to bed.", "marker": "that"},
                {"template": "So, {marker} decided to take a break.", "marker": "I"},
                {"template": "He finished his work, {marker} he went out with friends.", "marker": "then"},
                {"template": "First, {marker} will introduce myself.", "marker": "I"},
                {"template": "After {marker} long day, I just want to relax.", "marker": "a"},
                {"template": "He was tired, {marker} he went to sleep.", "marker": "so"},
                {"template": "First, {marker} eat breakfast, then I go to work.", "marker": "I"},
                {"template": "After {marker} the movie, we went for dinner.", "marker": "watching"},
            ]
        },
        "evidentials": {
            "true": [
                {"template": "{marker}, the prevalence of this phenomenon has increased significantly in recent decades.", "marker": "According to Smith (2023)"},
                {"template": "{marker}, the growth rate exhibits a clear exponential trajectory.", "marker": "As shown in Figure 3"},
                {"template": "Data from the recent survey {marker} a strong preference for remote work options among employees.", "marker": "indicates"},
                {"template": "The experimental results {marker} the efficacy of the proposed intervention.", "marker": "demonstrate"},
                {"template": "Research {marker} that early childhood education has long-term benefits.", "marker": "confirms"},
                {"template": "Studies {marker} a positive correlation between exercise and cognitive function.", "marker": "have found"},
                {"template": "The literature {marker} a consensus on the importance of interdisciplinary research.", "marker": "suggests"},
                {"template": "Empirical evidence {marker} the theoretical predictions.", "marker": "supports"},
                {"template": "As {marker} by previous studies, the effect is robust.", "marker": "reported"},
                {"template": "The analysis of the dataset {marker} a significant difference between the two groups.", "marker": "reveals"},
            ],
            "edge": [
                {"template": "{marker} my mother, the weather will be sunny tomorrow, so we should plan a picnic.", "marker": "According to"},
                {"template": "The sign {marker} that the road ahead is closed for construction.", "marker": "indicates"},
                {"template": "He {marker} that he would be late for the meeting.", "marker": "said"},
                {"template": "As {marker} by the weather forecast, it will rain tomorrow.", "marker": "predicted"},
                {"template": "The map {marker} the location of the treasure.", "marker": "shows"},
                {"template": "The doctor {marker} that I should rest.", "marker": "advised"},
                {"template": "The report {marker} a decline in sales.", "marker": "shows"},
                {"template": "The teacher {marker} that the exam would be difficult.", "marker": "warned"},
                {"template": "The data {marker} a problem with the sensor.", "marker": "suggests"},
                {"template": "The news {marker} a major breakthrough in science.", "marker": "reported"},
            ]
        }
    }

    context_templates = [
        "Previous research has extensively explored the topic of {topic}. Building upon this foundation, our current investigation delves into {specific_aspect}. ",
        "The field of {field} has witnessed rapid advancements in recent years, particularly concerning {sub_field}. This study aims to contribute to this growing body of knowledge by examining {novel_contribution}. ",
        "Understanding the complexities of {phenomenon} is crucial for {application}. While existing models offer valuable insights, they often overlook {limitation}. ",
        "A critical review of the literature reveals a persistent debate surrounding {controversial_topic}. This research seeks to provide empirical evidence to inform this discussion, focusing on {research_question}. ",
        "The implications of {concept} extend across various disciplines, from {discipline1} to {discipline2}. Our work specifically addresses its relevance within the context of {context_area}. ",
        "In light of recent developments in {technology}, it has become imperative to re-evaluate traditional approaches to {problem}. This paper presents a novel methodology designed to overcome these challenges. ",
        "The theoretical framework of {theory} has been widely applied in {domain}. However, its applicability to {new_domain} remains underexplored. ",
        "Addressing the challenges associated with {challenge} requires a multifaceted approach. This study integrates insights from {field1} and {field2} to propose a comprehensive solution. ",
        "The current understanding of {subject} is largely based on {previous_method}. This research introduces an alternative perspective, leveraging {new_method} to gain deeper insights. ",
        "The significance of {topic} in contemporary society cannot be overstated. This investigation contributes to a more nuanced understanding of its underlying mechanisms. ",
        "Recent studies have highlighted the importance of {factor} in {process}. Our research further elaborates on this by exploring the role of {sub_factor}. ",
        "The evolution of {system} has been a subject of intense academic inquiry. This paper examines the key drivers of its transformation, focusing on {key_driver}. ",
        "The interplay between {variable1} and {variable2} is a complex phenomenon with significant implications for {outcome}. This study provides empirical evidence to elucidate this relationship. ",
        "Developing effective strategies for {goal} is a pressing concern. This research proposes a data-driven approach to optimize resource allocation. ",
        "The application of {methodology} to {problem} has yielded promising results. This paper extends its utility by adapting it to a new context. ",
        "The debate surrounding {issue} has been ongoing for decades. This study offers a fresh perspective, drawing on {new_data_source} to inform the discussion. ",
        "Understanding the mechanisms underlying {process} is essential for designing targeted interventions. This research identifies key pathways involved in its regulation. ",
        "The impact of {phenomenon} on {population} has been widely documented. This study focuses on its long-term consequences, particularly for {sub_population}. ",
        "The theoretical foundations of {theory} provide a robust framework for analysis. This paper applies this framework to a novel dataset to test its predictive power. ",
        "Addressing the complexities of {problem} requires innovative solutions. This research explores the potential of {approach} to mitigate its adverse effects. "
    ]

    topics = ["artificial intelligence", "climate change", "quantum computing", "cognitive psychology", "sustainable development", "machine learning", "neuroscience", "economic policy", "social dynamics", "environmental science", "data analytics", "public health", "urban planning", "educational technology", "materials science", "biotechnology", "robotics", "cybersecurity", "renewable energy", "linguistics"]
    fields = ["computer science", "environmental studies", "physics", "psychology", "sociology", "engineering", "biology", "economics", "political science", "chemistry"]
    specific_aspects = ["its ethical implications", "the role of deep learning", "its impact on society", "the underlying neural mechanisms", "policy interventions", "algorithmic bias", "human-computer interaction", "resource optimization", "data privacy concerns", "the future of work"]
    novel_contributions = ["a novel predictive model", "a new theoretical framework", "an empirical validation of the hypothesis", "a comprehensive review of existing literature", "a comparative analysis of different approaches", "a case study of its real-world application", "a methodological innovation", "a deeper understanding of the causal mechanisms", "a set of policy recommendations", "a new dataset for future research"]
    limitations = ["its limited generalizability", "the lack of longitudinal data", "its inability to account for individual variability", "the computational complexity", "the ethical considerations", "its reliance on simplified assumptions", "the absence of real-world validation", "its susceptibility to bias", "the difficulty in interpreting its outputs", "the scalability challenges"]
    controversial_topics = ["the ethics of AI", "the role of government in the economy", "the impact of social media on mental health", "the origins of consciousness", "the effectiveness of different educational reforms", "the causes of climate change", "the future of work", "the regulation of emerging technologies", "the nature of free will", "the relationship between technology and society"]
    research_questions = ["its long-term effects", "the mediating factors", "the moderating variables", "its cross-cultural applicability", "the underlying psychological processes", "its economic consequences", "the social implications", "the environmental impact", "the policy implications", "the ethical considerations"]
    concepts = ["big data", "blockchain", "virtual reality", "augmented reality", "nanotechnology", "CRISPR", "neuroplasticity", "game theory", "complex systems", "network theory"]
    application_areas = ["healthcare", "finance", "education", "manufacturing", "agriculture", "transportation", "urban development", "environmental conservation", "social policy", "scientific discovery"]
    technologies = ["machine learning", "blockchain", "IoT", "robotics", "biotechnology", "quantum computing", "AI", "virtual reality", "nanotechnology", "CRISPR"]
    problems = ["resource scarcity", "climate change mitigation", "disease prevention", "sustainable energy", "data security", "urban congestion", "educational inequality", "economic instability", "social polarization", "environmental degradation"]
    theories = ["general relativity", "evolutionary theory", "cognitive dissonance", "social learning theory", "chaos theory", "game theory", "attachment theory", "diffusion of innovations", "systems theory", "constructivism"]
    domains = ["physics", "biology", "psychology", "sociology", "economics", "political science", "education", "engineering", "computer science", "environmental science"]
    challenges = ["data scarcity", "model interpretability", "scalability", "ethical considerations", "bias in algorithms", "resource constraints", "interdisciplinary collaboration", "public acceptance", "regulatory frameworks", "technological limitations"]
    fields_for_integration = ["computer science", "linguistics", "psychology", "sociology", "economics", "biology", "physics", "chemistry", "environmental science", "political science"]
    new_methods = ["deep learning", "reinforcement learning", "causal inference", "network analysis", "agent-based modeling", "natural language processing", "computer vision", "genomic sequencing", "neuroimaging", "qualitative research"]
    subjects = ["human cognition", "social behavior", "ecosystem dynamics", "economic systems", "political processes", "language acquisition", "material properties", "biological processes", "urban development", "climate patterns"]
    previous_methods = ["traditional statistical analysis", "qualitative interviews", "survey research", "experimental designs", "observational studies", "simulation models", "case studies", "literature reviews", "theoretical frameworks", "expert opinions"]
    factors = ["socioeconomic status", "cultural background", "technological adoption", "policy interventions", "environmental conditions", "individual differences", "organizational structure", "market dynamics", "psychological traits", "biological markers"]
    processes = ["learning", "decision-making", "social interaction", "economic growth", "ecosystem resilience", "disease progression", "technological innovation", "urbanization", "language development", "cognitive development"]
    sub_factors = ["peer influence", "family support", "educational attainment", "access to resources", "media consumption", "genetic predispositions", "neurological pathways", "cultural norms", "institutional policies", "individual agency"]
    systems = ["ecosystems", "economic systems", "social networks", "biological systems", "cognitive systems", "political systems", "technological systems", "urban systems", "educational systems", "healthcare systems"]
    key_drivers = ["technological advancements", "globalization", "demographic shifts", "policy changes", "environmental pressures", "economic forces", "social movements", "cultural shifts", "scientific discoveries", "resource availability"]
    variable_pairs = [("income", "education"), ("exercise", "cognitive function"), ("social media use", "mental health"), ("temperature", "crop yield"), ("policy stringency", "disease spread"), ("urban density", "air quality"), ("biodiversity", "ecosystem stability"), ("economic inequality", "social unrest"), ("technological adoption", "productivity growth"), ("sleep quality", "academic performance")]
    outcomes = ["well-being", "productivity", "health outcomes", "environmental sustainability", "economic development", "social cohesion", "technological progress", "educational attainment", "public health", "urban resilience"]
    methodologies = ["quantitative analysis", "qualitative research", "mixed methods", "experimental design", "survey research", "case study", "ethnography", "content analysis", "discourse analysis", "meta-analysis"]
    new_contexts = ["emerging markets", "remote work environments", "online learning platforms", "virtual communities", "developing countries", "post-pandemic societies", "digital ecosystems", "global supply chains", "smart cities", "personalized medicine"]
    issues = ["climate change", "income inequality", "access to healthcare", "educational disparities", "food security", "water scarcity", "cybersecurity threats", "data privacy", "social justice", "political polarization"]
    new_data_sources = ["social media data", "satellite imagery", "genomic data", "sensor data", "electronic health records", "online forums", "mobile phone data", "wearable device data", "citizen science data", "archival records"]
    mechanisms = ["neural pathways", "molecular interactions", "social norms", "economic incentives", "cognitive biases", "feedback loops", "regulatory processes", "communication patterns", "learning algorithms", "environmental factors"]
    populations = ["adolescents", "elderly", "immigrants", "minority groups", "students", "healthcare professionals", "policymakers", "consumers", "employees", "researchers"]
    sub_populations = ["urban youth", "rural elderly", "first-generation immigrants", "STEM students", "frontline workers", "small business owners", "digital natives", "low-income communities", "patients with chronic diseases", "early career researchers"]
    approaches = ["community-based interventions", "technological solutions", "policy reforms", "educational programs", "behavioral nudges", "economic incentives", "public awareness campaigns", "infrastructure development", "international cooperation", "research and development"]
    adverse_effects = ["environmental degradation", "social inequality", "health disparities", "economic instability", "technological unemployment", "privacy violations", "cyberbullying", "misinformation spread", "resource depletion", "biodiversity loss"]

    def get_random_context_fillers():
        return {
            "topic": random.choice(topics),
            "specific_aspect": random.choice(specific_aspects),
            "field": random.choice(fields),
            "sub_field": random.choice(fields),
            "novel_contribution": random.choice(novel_contributions),
            "phenomenon": random.choice(topics),
            "application": random.choice(application_areas),
            "limitation": random.choice(limitations),
            "controversial_topic": random.choice(controversial_topics),
            "research_question": random.choice(research_questions),
            "concept": random.choice(concepts),
            "discipline1": random.choice(fields),
            "discipline2": random.choice(fields),
            "context_area": random.choice(application_areas),
            "technology": random.choice(technologies),
            "problem": random.choice(problems),
            "theory": random.choice(theories),
            "domain": random.choice(domains),
            "challenge": random.choice(challenges),
            "field1": random.choice(fields_for_integration),
            "field2": random.choice(fields_for_integration),
            "new_method": random.choice(new_methods),
            "subject": random.choice(subjects),
            "previous_method": random.choice(previous_methods),
            "factor": random.choice(factors),
            "process": random.choice(processes),
            "sub_factor": random.choice(sub_factors),
            "system": random.choice(systems),
            "key_driver": random.choice(key_drivers),
            "variable1": random.choice([v[0] for v in variable_pairs]),
            "variable2": random.choice([v[1] for v in variable_pairs]),
            "outcome": random.choice(outcomes),
            "methodology": random.choice(methodologies),
            "new_domain": random.choice(domains),
            "new_context": random.choice(new_contexts),
            "issue": random.choice(issues),
            "new_data_source": random.choice(new_data_sources),
            "mechanisms": random.choice(mechanisms),
            "population": random.choice(populations),
            "sub_population": random.choice(sub_populations),
            "approach": random.choice(approaches),
            "adverse_effects": random.choice(adverse_effects),
            "goal": random.choice(outcomes) # Added 'goal' here
        }

    generated_texts = set()

    while len(data) < num_rows:
        category = random.choice(list(sentence_templates.keys()))
        is_true_marker = random.choice([True, False])
        template_type = "true" if is_true_marker else "edge"

        selected_template_info = random.choice(sentence_templates[category][template_type])
        sentence_template = selected_template_info["template"]
        marker_text = selected_template_info["marker"]

        # Generate target sentence
        target_sentence = sentence_template.format(marker=marker_text)

        # Generate context
        pre_context_template = random.choice(context_templates)
        post_context_template = random.choice(context_templates)

        pre_context = pre_context_template.format(**get_random_context_fillers())
        post_context = post_context_template.format(**get_random_context_fillers())

        full_text = f"{pre_context}{target_sentence} {post_context}"

        # Ensure uniqueness
        if full_text in generated_texts:
            continue
        generated_texts.add(full_text)

        # Determine reason for marker classification
        reason = ""
        if is_true_marker:
            if category == "self_mentions":
                reason = "Academic self-reference (e.g., 'we argue', 'our research')."
            elif category == "hedges":
                reason = "Epistemic modal or cautious phrasing in academic context."
            elif category == "boosters":
                reason = "Strong assertion or certainty in academic discourse."
            elif category == "frame_markers":
                reason = "Discourse organizer or structural indicator."
            elif category == "code_glosses":
                reason = "Clarification, exemplification, or reformulation."
            elif category == "engagement_markers":
                reason = "Direct address to reader, guiding attention or thought."
            elif category == "transitions":
                reason = "Logical connector between ideas or arguments."
            elif category == "evidentials":
                reason = "Reference to source of information or evidence."
        else:
            if category == "self_mentions":
                reason = "Personal narrative or non-academic possessive/action."
            elif category == "hedges":
                reason = "Deontic modal, casual expression, or non-epistemic use."
            elif category == "boosters":
                reason = "Casual intensifier or direct instruction."
            elif category == "frame_markers":
                reason = "Temporal sequence in narrative or non-structural use."
            elif category == "code_glosses":
                reason = "General comparison, causal connector, or non-explanatory use."
            elif category == "engagement_markers":
                reason = "Direct personal address or conversational question."
            elif category == "transitions":
                reason = "Temporal sequence in narrative or non-logical connection."
            elif category == "evidentials":
                reason = "Personal source, literal meaning, or non-academic report."

        data.append({
            "sentence_id": str(uuid.uuid4()), # Unique ID for each entry
            "text": full_text.strip(),
            "marker_text": marker_text,
            "marker_category": category,
            "is_metadiscourse": is_true_marker,
            "reason": reason
        })
        if len(data) % 1000 == 0:
            print(f"Generated {len(data)} rows...")

    return data

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_dataset(num_rows=10000) # You can increase this number
    
    # Convert to pandas DataFrame and save as CSV
    df = pd.DataFrame(dataset)
    csv_output_file = "synthetic_metadiscourse_dataset.csv"
    df.to_csv(csv_output_file, index=False, encoding="utf-8")
    print(f"Dataset generated and saved to {csv_output_file}")

    # Optional: Print a sample to verify structure
    print("\n--- Sample of generated data (first 2 entries) ---")
    print(df.head(2).to_string())
