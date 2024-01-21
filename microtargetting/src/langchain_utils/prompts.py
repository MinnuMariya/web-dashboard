
expert1_prompts = {
    "Content Tailored to Specific Groups" : """
        Analyze the following tweet to determine if it targets specific groups and provide the answer in a short paragraph:

        Tweet Text: {tweet}

        Assess:

        1. Does the content seem to target specific interests, beliefs, or demographic characteristics?
        2. How does the tweet's language and tone align with particular voter groups?
    """,
    "Use of Personal Data" : """
        Evaluate the tweet for its use of personal data in targeting voters and provide the answer in a short paragraph:

        Tweet Text: {tweet}

        Consider:

        1. Does the tweet align with specific personal interests or behaviors?
        2. How might the tweet reflect known political leanings or personal preferences?
    """,
    "Sophisticated Data Integration" : """
        Analyze the tweet for evidence of sophisticated data integration and and provide the answer in a short paragraph:

        Tweet Text: {tweet}

        Questions:

        1. Does the tweet appear to be part of a system that integrates various types of user data?
        2. Are there indications that the tweet is a component of a larger, data-driven campaign strategy?
    """,
    "Selective Messaging Based on User Behavior" : """
        Assess the tweet for selective messaging based on user behavior and and provide the answer in a short paragraph:

        Tweet Text: {tweet}

        Evaluate:

        1. How does the tweet align with specific user behaviors or preferences indicated on social media profiles?
        2. Are there patterns in the content that suggest targeting based on observed online activities?
    """,
    "Customization Based on Region or Specific Interests" : """
        Determine if the tweet is customized for specific regions or interests and provide the answer in a short paragraph:

        Tweet Text: {tweet}

        Look for:

        1. Does the tweet cater to the preferences or interests specific to a certain region?
        2. Are there elements in the tweet that align with region-specific political or cultural trends?
    """
}

expert2_prompts = """
    Evaluate the tweet for potential microtargeting:

    Tweet Text: {tweet}

    Consider the following criteria:

    1. Personality Congruence: Does the tweet align with specific personality traits (thinking vs. feeling)?
    2. Ad Appeal: Analyze if the tweet contains rational or emotional appeals, and if they match a particular personality type.
    3. Perceived Relevance: Assess whether the tweet appears crafted to be particularly relevant to individual interests or beliefs.
    4. Political Content: Determine if the political messaging in the tweet is tailored to resonate with certain personality types.
    5. User Engagement: Look at the engagement levels - does the tweet seem to resonate more with users of a specific personality type?
    
    Based on these criteria, provide a reasoned assessment of the likelihood that this tweet is being used for microtargeting.
"""

prediction_prompt = """
    Two experts have provided opinions regarding a specific tweet's potential for microtargeting. The tweet in question is as follows:

    Tweet Text: {tweet}

    Expert 1's Opinion: {expert1_opinion}
    Expert 2's Opinion: {expert2_opinion}

    Considering the analyses provided by Expert 1 and Expert 2, assess the possibility of microtargeting in the tweet's content and reason for your assesment.
    In the answer dont mention Expert 1 or Expert 2.

    Expert 1 suggests [Summarize Expert 1's analysis]. Expert 2 suggests [Summarize Expert 2's analysis].

    Assessment: [Assess the possibility of microtargeting] 
    Reason: [Reason for your assesment]
"""