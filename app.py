import streamlit as st
from crewai import Agent, Task, Crew
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os


os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_MODEL_NAME"] = ""

# Streamlit configuration
st.set_page_config(
    page_title="SuperCharged ITSM - Intelligent Agents in ITSM",
    page_icon="⚙️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown(
    """
    <style>
    .main-header {font-size: 30px; color: #0A2E73; text-align: center; font-weight: bold;}
    .section-header {font-size: 24px; color: #1E3A8A; font-weight: bold;}
    .subheader {font-size: 20px; color: #4B5563; margin-top: 20px;}
    .input-box {background-color: #F3F4F6; padding: 10px; border-radius: 10px;}
    .recommendation-box {background-color: #E0F2FE; padding: 15px; border-radius: 10px; font-size: 18px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Main Header
st.markdown("<div class='main-header'>SuperCharged ITSM - Intelligent Agents in ITSM</div>", unsafe_allow_html=True)


sample_incidents = [
    {"Incident ID": "IMXXXXXXXXXXXX", "Summary": "AXT | Unable to access application", "Assigned Group": "Axia Application Support", "Resolution": "Verified user credentials in AD. Checked sign-in logs; confirmed correct login. User accessed application after retry."},
    {"Incident ID": "IMXXXXXXXXXXXX", "Summary": "AXT | SYSTEM NOT UP", "Assigned Group": "Axia Application Support", "Resolution": "System accessible. Monitoring ongoing."},
    {"Incident ID": "IMXXXXXXXXXXXX", "Summary": "One of the services inaccessible due to memory issue", "Assigned Group": "CMP Support", "Resolution": "Allocated more resources to the affected system."},
    {"Incident ID": "IMXXXXXXXXXXXX", "Summary": "INFO INACCESSIBLE", "Assigned Group": "CXI Support", "Resolution": "System working fine now. For issues with specific services, provide service ID for investigation."},
    {"Incident ID": "IMXXXXXXXXXXXX", "Summary": "AXT | SERVER DOWN | Server IP Address: 10.X.X.X", "Assigned Group": "Server Support", "Resolution": "IT Engineering: Pending patches process completed. Issue resolved after patches update."},
    {"Incident ID": "IMXXXXXXXXXXXX", "Summary": "AXT | Server Down | Unable to login web-based email | User X", "Assigned Group": "Server Support", "Resolution": """Findings/Troubleshooting:
        1. Validated user details in AD; observed logon name change.
        2. Corrected AD details.
        3. Requested appointment for further support.
        4. User successfully accessed email after session.
        5. Assisted with password change during support session.
        6. User successfully logged into web email."""},
    {"Incident ID": "IMXXXXXXXXXXXX", "Summary": "AXT | Security | Windows login attempt (Unable to login) - Case# XXXX | SOC", "Assigned Group": "Server Support", "Resolution": "User changed password. Successful sign-in log observed."},
    {"Incident ID": "IMXXXXXXXXXXXX", "Summary": "AXT | Printer Issue - unable to access", "Assigned Group": "Server Support", "Resolution": "SSL certificate renewed and patches updated. Issue resolved."}
]


def select_relevant_incidents(summary, sample_incidents, top_n=5, similarity_threshold=0.2, similarity_weight=0.2, keyword_weight=0.2):
    # Filter incidents by assigned workgroup
    filtered_incidents = [incident for incident in sample_incidents]
    print(filtered_incidents)

    # Perform semantic search using cosine similarity
    texts = [summary] + [incident["Summary"] for incident in filtered_incidents]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Keyword-based matching score
    summary_keywords = set(summary.lower().split())
    keyword_scores = []
    for incident in filtered_incidents:
        incident_keywords = set(incident["Summary"].lower().split())
        common_keywords = summary_keywords.intersection(incident_keywords)
        keyword_score = len(common_keywords) / len(summary_keywords)  # Jaccard-like score
        keyword_scores.append(keyword_score)

    # Calculate hybrid relevance score
    relevance_scores = [
        (similarity_weight * cosine_similarities[i]) + (keyword_weight * keyword_scores[i])
        for i in range(len(filtered_incidents))
    ]

    # Filter results based on similarity threshold for semantic similarity
    relevant_indices = [i for i in range(len(relevance_scores)) if cosine_similarities[i] >= similarity_threshold]

    # Sort by relevance score and return the top_n results
    sorted_relevant_indices = sorted(relevant_indices, key=lambda i: relevance_scores[i], reverse=True)[:top_n]
    print(sorted_relevant_indices)
    return [filtered_incidents[i] for i in sorted_relevant_indices]

# Agents for workgroup assignment and recommendation
workgroup_agent = Agent(
    role="Workgroup Assigner",
    goal="Select one suitable workgroup based on the assigned workgroups list. Default to L1_ServiceDesk if you are unsure. Keep the respond short and concise.",
    backstory="An experienced support coordinator specializing in making confident and precise workgroup assignments based on the provided data."
)

recommendation_agent = Agent(
    role="Recommendation Provider",
    goal="Provide specific recommendations based on similar past issues and resolutions based the assigned workgroup.",
    backstory="A support analyst with access to historical resolutions."
)

reporting_agent = Agent(
    role="Reporting Agent",
    goal="Summarize and generate useful insights of the output of the workgroup assignment and recommendation agents.",
    backstory="An assistant focused on consolidating information and generate useful insights for efficient incident management."
)

# UI for Incident Submission
st.markdown("### Submit a New Incident Ticket")
user_id = st.text_input("Incident ID", placeholder="Enter Incident ID")
summary = st.text_area("Describe the issue:", placeholder="Provide a brief summary of the issue")

# Define the task for workgroup assignment and display results
if st.button("Find Suitable Workgroup"):
    relevant_incidents = select_relevant_incidents(summary, sample_incidents)
    relevant_text = "\n".join([f"- **Summary**: {incident['Summary']}\n  **Assigned Group**: {incident['Assigned Group']}\n  **Resolution**: {incident['Resolution']}" for incident in relevant_incidents])

    assign_workgroup_task = Task(
        description=f"Select only one suitable workgroup based on incident summary and past incidents:\n{relevant_text}",
        agent=workgroup_agent,
        expected_output="markdown"
    )

    with st.spinner("Assigning workgroup..."):
        support_crew = Crew(agents=[workgroup_agent], tasks=[assign_workgroup_task], verbose=True)
        results = support_crew.kickoff(inputs={"incident_id": user_id, "summary": summary})

    # Store workgroup assignment result and relevant incident data in session
    st.session_state["workgroup_result"] = results.tasks_output[0].raw.strip()
    st.session_state["relevant_incidents_text"] = relevant_text

    # Display assigned workgroup
    st.markdown("#### Assigned Workgroup")
    st.markdown(f"<div class='recommendation-box'>{st.session_state['workgroup_result']}</div>", unsafe_allow_html=True)

# Run the recommendation agent with the same incident data used in workgroup assignment
if "relevant_incidents_text" in st.session_state and st.button("Get Recommendation"):
    recommendation_task = Task(
        description="Based on the following past incident resolutions related to the assigned workgroup:\n\n"
                    f"{st.session_state['relevant_incidents_text']}\n\n"
                    "Generate an actionable recommendation for resolving similar issues. The recommendation should "
                    "consider common solutions used in these incidents and be specific to the assigned workgroup.",
        agent=recommendation_agent,
        expected_output="markdown"
    )

    # Run the recommendation task
    with st.spinner("Generating recommendation..."):
        support_crew = Crew(agents=[recommendation_agent], tasks=[recommendation_task], verbose=True)
        recommendation_result = support_crew.kickoff(inputs={"summary": summary})

    # Display the recommendation
    st.markdown("#### Recommendation")
    st.markdown(f"<div class='recommendation-box'>{recommendation_result.tasks_output[0].raw}</div>", unsafe_allow_html=True)

    # Store the recommendation result
    st.session_state["recommendation_result"] = recommendation_result.tasks_output[0].raw

# Reporting Agent
if "workgroup_result" in st.session_state and "recommendation_result" in st.session_state and st.button("Generate Report"):
    report_task = Task(
        description=f"Summarize and generate insightful info of the workgroup assignment '{st.session_state['workgroup_result']}' and recommendation '{st.session_state['recommendation_result']}' for incident {user_id}.",
        agent=reporting_agent,
        expected_output="markdown"  # Specify expected output format here
    )

    with st.spinner("Generating report..."):
        support_crew = Crew(agents=[reporting_agent], tasks=[report_task], verbose=True)
        report_result = support_crew.kickoff()

    # Display the report
    st.markdown("#### Report")
    st.markdown(f"<div class='recommendation-box'>{report_result.tasks_output[0].raw}</div>", unsafe_allow_html=True)
