# Import Dependencies
##############################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import sys
import os
from PIL import Image
# import pickle
# # model = pickle.load(open('model.sav', 'rb'))
from warnings import filterwarnings
filterwarnings('ignore')



@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")


# --- Sidebar Navigation ---
img = Image.open('imgs/sidebar3.png')
st.sidebar.image(img)
page = st.sidebar.radio("📑 Pages👇", ["🏠 Home", "📊 Analysis", "😎 Prediction"])

# --- Page 1: Home ---
if page == "🏠 Home":
    ###############################################################################################
    st.title('Attention Span Classifier')
    st.write('Developed by `Mehedi Hasan`, `Statistical Analyst & Data Scientist`')
    st.write('`B.Sc.` in `CSE` & `M.Sc.` in `Applied Statistics & Data Science`, `JU`')
    st.write('`Email: azadmehedi12121@gmail.com` | `Phone: +88 01612 956266`')
    img = Image.open("imgs/attentionspan.png")  
    st.image(img)


    st.markdown("""
    ### 📖 Project Overview
    This project is developed based on my `M.Sc. thesis` titled 
    #####  `"An Assessment of Attention Span among College Students in Dhaka: Patterns, Potential Implications, and Influencing Factors"`
    """)
    '---'


    col1, col2, col3, col4, col5 = st.columns(5)

    # Abstract
    with col1:
        abs = st.checkbox('Abstract')
    if abs:
        st.write('#### `Abstract`')
        st.write('''In today’s digital age, maintaining attention has become increasingly challenging for students due to digital distractions around them. This study, titled “An Assessment of Attention Span among College Students in Dhaka: Patterns, Potential Implications, and Influencing Factors,” explores the attention span of college students across urban and suburban areas of Dhaka, Bangladesh. Using primary data collected from 349 students through a structured questionnaire, the study employed descriptive statistics, hypothesis testing, and machine learning techniques to discover key behavioral, psychological, and digital factors influencing student attention. The analysis revealed that the average attention span was approximately 28.7 minutes, with significant variation linked to screen time, study habits, mental fatigue, and anxiety. Statistical tests confirmed strong associations between digital behaviors and reduced focus. To classify students into High, Medium, and Low attention categories, an XGBoost classification model was developed, achieving 81.08% accuracy. Key predictive features included max continuous reading time, study hours at home, and digital and mental well-being indicators. The findings show the critical impact of digital lifestyle and mental health on academic focus. Practical recommendations include promoting digital mindfulness, structured study routines, and institutional support for cognitive well-being. This research offers timely insights into student attention patterns in a developing country context and provides a foundation for future educational interventions and policy reforms aligned with the realities of digital learning.
        \n Keywords: Attention Span, College Students, Digital Behavior, XGBoost, Educational Psychology, Bangladesh
        '''
    )
    '---'
    
    # Significance & Background of the study
    with col2:
        significance_Background = st.checkbox('Significance & Background')
    if significance_Background:
        st.write('### `Significance & Background of the study`')
        st.write('''With the limited local research, this study is important to explore, finding out the causes of digital distraction affects college students of Dhaka, Bangladesh. In this critical situation, understanding attention span is vital because it impacts learning, academic performance & mental health.
        \n The findings will help directly parents to care their boys-girls accordingly, teachers will planning & develop their teaching accordingly and policymakers to create balanced digital education policies.
        ''')
        st.write('### `Background of the study`')
        st.write('''
                - Students Attention span plays a crucial role in academic performance that lead a successful career & shines in persona life. 
                - Attention span of students are shrinking, both locally & globally.
                - Digital devices dependency, bad influence of social media, notification, reels plays a vital role to decrease of attention span.
                - Dhaka as the capital city has a high density of college students with diverse background.
                 ''')
        '---'
        


    # Research Objectives & Questions
    with col3:
        objectives_questions = st.checkbox('Objectives & Questions')
    if objectives_questions:
        st.write('### `Research Objectives`')
        st.write('''
                1.	To `assess` the attention span among `college students of Dhaka`.
                2.  To identify the `patterns & influencing factors` of attention span.
                3.	To `classify students` based on `attention category` `(High, Medium, Low)`
                4.	Suggesting `potential implications` on academic performance. 
                 ''')
        st.write('### `Research Questions`')
        st.write('''
                1.	What is the current attention span of college students in urban and suburban areas of Bangladesh during academic activities?
                2.	How do digital device usage patterns (including social media, short videos, and screen time) affect students’ attention span?
                3.	What physiological and psychological factors (e.g., mental fatigue, anxiety, and sleep quality) influence students’ attention span?
                4.	To what extent do external distractions (such as family/friends' pressure, notifications, and study environment) impact students’ ability to focus?
                5.	Are there significant differences in attention span related to demographic variables such as age, gender, education level, and institution location?
                6.	How effective are students’ self-regulation strategies (like reducing screen time, taking breaks, following routines) in improving attention span?
                7.	What are the main distractions affecting students during study, and how frequently do they experience unintentional inattention?
                8.	How do students perceive the change in their attention span over time, especially in the digital age?
                 ''')
        '---'
        
    # Dataset & Hypothesis testing
    with col4:
        dataset_hypothesis = st.checkbox('Dataset & Hypothesis')
    if dataset_hypothesis:
        st.write('### `Dataset Information - (Primary Dataset)`')
        st.write('''
                - Source : Data collected through both `structured printed questionnaire` & `google form`.
                - Sampling Technique : `Non-Probability Convenience Sampling` Technique.
                - Dataset Details: 
                    - `349 rows`
                    - `31 columns`
                 ''')
        st.write('### `Hyothesis Testing`')
        st.write('''
                - Conducted `16 Hypothesis Tests`.
                - Test Used: 
                    - `Independent Samples t-test`
                    - `One-way ANOVA (Analysis of Variance)`
                    - `Chi-square Test of Independence`
                    - `Pearson Correlation Coefficient`
                 ''')   
        '---'

    
    # Model Building
    with col5:
        model = st.checkbox('Model Building')
    if model:
        st.write('### `Model Building - XGBoost Classifier`')
        st.write('''
            - Model Selection: `XGBost Classifier`
            - Target variable: A new column `“attention_category”` was created by `categorizing the numeric variable` `“average_attention_span” `
            into `three groups.`
                - `High: (score>30 min)`
                - `Medium: (score 11-30 min)`
                - `Low: (score<=10 min)`
                - This setup `excluded` the direct use of `average_attention_span as a predictor`, ensuring a more `reliable and unbiased` classification outcome.

            - Feature Engineering: 
                - `Label Encoding` (for `converting Categorical data into Numerical`)
                - `Standard Scaler` (for `Scaling Numeric data`)
                - `SMOTE` (for `handing class imbalance`)

            - Model Training: 
                - The dataset was `split` using `stratified sampling` to maintain `class balance` in both `training & testing set`.
                - `Training set`: `80%` of the data
                - `Testing set`: `20%` of the data
        ''')

    # Figures
    



#################################################################################
#################################################################################
# --- Page 2: Analysis---
elif page == "📊 Analysis":
    st.title("📊 EDA & Statistical Tests")

    # Import Dependencies
    ##############################################
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    # import seaborn as sns
    import streamlit as st
    import sys
    import os
    from PIL import Image
    import pickle
    # model = pickle.load(open('model.sav', 'rb'))
    from warnings import filterwarnings
    filterwarnings('ignore')

    ###############################################################################################
    # st.title('Attention Span Classifier')
    # st.write('Created by `Mehedi Hasan`, `Statistical Analyst & Data Scientist`')
    # st.write('`B.Sc.` in `CSE` & `M.Sc.` in `Applied Statistics & Data Science`, `JU`')

    ###############################################################################################
    # upload an image
    img = Image.open("imgs/image.png")  
    st.image(img)
    st.write('---')


    ###############################################################################################
    # Load the Dataset
    # df = pd.read_csv('D:/Mehedi Azad/Streamlit/attention_span/cleaned_dataset.csv')
    df = load_data()
    # basic information about dataset
    rows = df.shape[0]
    cols = df.shape[1]
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    num_count = len(num_cols)
    cat_count = len(cat_cols)
    missing_count = df.isna().sum().sum()
    duplicate_count = df.duplicated().sum()


    ########################################################################
    st.subheader("📊 `Dataset KPIs`")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Rows", rows)
    with col2:
        st.metric("Columns", cols)
    with col3:
        st.metric("Numeric Cols", num_count)
    with col4:
        st.metric("Categorical Cols", cat_count)
    with col5:
        st.metric("Missing values", missing_count)
    with col6:
        st.metric("Duplicate values", duplicate_count)
    # st.write('---')


    ########################################################################
    # unique values of the dataset
    st.subheader('`Unique values of the dataset columns`')
    # unique_df = st.checkbox('✅Unique values of the dataset columns')
    unique_values = df[df.columns].nunique()
    unique_values_df = pd.DataFrame(unique_values).T
    st.write(unique_values_df)
    # if unique_df:
    #     st.write(unique_values_df)
    st.write('---')





    ###############################################################################################
    # Filtering operations for filterable dataset
    show_filters = st.sidebar.checkbox('`Show Filtering Options`')
    filters = {}
    if show_filters:
        st.sidebar.markdown('## 🔎Filtering Options')

        # Numeric + Categorical columns একসাথে multiselect filter
        all_cols = num_cols + cat_cols
        filters = {}

        for col in all_cols:
            unique_vals = sorted(df[col].dropna().unique().tolist())  # ইউনিক ভ্যালু
            filters[col] = st.sidebar.multiselect(
                f"{col}", 
                options=unique_vals, 
                default=unique_vals  # ডিফল্টে সব সিলেক্ট করা থাকবে
            )

    # Apply filters on dataset
    filtered_df = df.copy()
    for col, selected_values in filters.items():
        if selected_values:  # multiselect ফাঁকা না হলে
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]


    ###############################################################################################
    # checkbox of showing the dataset
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('💾`Dataset`')
        show_dataset = st.checkbox('show Dataset')
    if show_dataset:
        st.dataframe(df)
    with col2:
        st.subheader('💾`Filterable Dataset`')
        show_dataset = st.checkbox('Show filterable Dataset')
    if show_dataset:
        st.dataframe(filtered_df)
    '---'



    #################################################################################################
    # Descriptive analysis of Numerical Columns
    univariate_stat = {}
    for col in num_cols:
        univariate_stat[col] = {
            'count': df[col].count(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            '25%': df[col].quantile(0.25),
            '50% (median)': df[col].median(),
            '75%': df[col].quantile(0.75),
            'max': df[col].max()
        }
    numeric_stats = pd.DataFrame(univariate_stat).T

    # Descriptive analysis of Categorical Columns
    univariate_stat = {}
    for col in cat_cols[:-1]:  # Adjusting to avoid index error on last categorical column
        univariate_stat[col] = {
            'Unique categories': df[col].nunique(),
            'frequency': df[col].mode().values[0],
            'Percentage Distribution': df[col].value_counts(normalize=True) * 100,
        }
    categorical_stats = pd.DataFrame(univariate_stat).T


    # unique values of the dataset
    unique_values = df[df.columns].nunique()
    unique_values_df = pd.DataFrame(unique_values).T


    st.subheader('📝`Descriptive Analysis of Features`')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        show_num_stat = st.checkbox('Numeric Features Stats')
    if show_num_stat:
        st.write('`Numeric Features Statistics`')
        st.write(numeric_stats)

    with col2:
        show_cat_stat = st.checkbox('Categorical Features Stats')
    if show_cat_stat:
        st.write('`Categorical Features Statistics`')
        st.write(categorical_stats)

    with col3:  
        unique_df = st.checkbox('Unique values of Columns')
    if unique_df:
        st.write('`Unique values of Columns`')
        st.write(unique_values_df)

    with col4:
        n_cols = st.checkbox('Numeric Columns')
    if n_cols:
        st.write('`Numeric Columns List`')
        st.write(num_cols)

    with col5:
        c_cols = st.checkbox('Categorical Columns')
    if c_cols:
        st.write('`Categorical Columns List`')
        st.write(cat_cols)
    st.write('---')
        


    ###############################################################################################
    # Chi-Square test results
    data = {
        "Variable Name": [
            "gender", "institution_location", "age", "daily_digital_device_usage",
            "most_used_digital_platform", "watch_short_videos", "have_social_media_account",
            "mental_fatigue_frequency", "pre_study_anxiety", "regular_sufficient_sleep",
            "digital_device_stress", "biggest_study_distraction", "avoid_notifications",
            "family_friends_pressure_effect", "follow_study_routine", "tried_reduce_screen_time",
            "practice_meditation", "study_break_frequency", "max_continuous_reading_time",
            "unintentional_inattention", "attention_span_decreased"
        ],
        "Categories": [
            "Female, Male, Other",
            "City, Suburban, Village (Rural)",
            "16, 17, 18, 19, 20",
            "0, 2, 4, 6, 8",
            "Facebook, Instagram, Others, TV, TikTok, Twitter, YouTube",
            "No, Sometimes, Yes",
            "Yes",
            "Never, Often, Rarely, Sometimes",
            "Never, Often, Rarely, Sometimes",
            "No, Sometimes, Yes",
            "No, Not sure, Yes",
            "Daydreams, Noise, Notifications, Others",
            "No, Sometimes, Yes",
            "No, Sometimes, Yes",
            "No, Yes",
            "No, Sometimes, Yes",
            "No, Sometimes, Yes",
            "0, 15, 30, 60",
            "3, 10, 15, 30",
            "No, Sometimes, Yes",
            "No, Not sure, Yes"
        ],
        "Chi-square Statistic": [
            23.27, 7.05, 17.31, 118.73, 51.78, 36.08, 0.00, 41.71, 34.99,
            7.72, 10.96, 21.32, 28.06, 10.54, 11.82, 28.20, 8.06, 68.80,
            110.68, 20.48, 18.35
        ],
        "p-value": [
            0.0098, 0.7208, 0.6329, 0.0000, 0.0080, 0.0001, 1.0000, 0.0002,
            0.0025, 0.6558, 0.3607, 0.1268, 0.0018, 0.3949, 0.0373, 0.0017,
            0.6230, 0.0000, 0.0000, 0.0251, 0.0493
        ],
        "Significant Association?": [
            "✅ Yes (Reject H₀)", "❌ No (Fail to Reject H₀)", "❌ No (Fail to Reject H₀)",
            "✅ Yes (Reject H₀)", "✅ Yes (Reject H₀)", "✅ Yes (Reject H₀)", "❌ No (Fail to Reject H₀)",
            "✅ Yes (Reject H₀)", "✅ Yes (Reject H₀)", "❌ No (Fail to Reject H₀)",
            "❌ No (Fail to Reject H₀)", "❌ No (Fail to Reject H₀)", "✅ Yes (Reject H₀)",
            "❌ No (Fail to Reject H₀)", "✅ Yes (Reject H₀)", "✅ Yes (Reject H₀)",
            "❌ No (Fail to Reject H₀)", "✅ Yes (Reject H₀)", "✅ Yes (Reject H₀)",
            "✅ Yes (Reject H₀)", "✅ Yes (Reject H₀)"
        ]
    }
    chi_df = pd.DataFrame(data)

    # significant Trend Analysis
    data = {
        "Variable": [
            "Gender",
            "Daily Digital Device Usage",
            "Watch Short Videos",
            "Mental Fatigue",
            "Pre-study Anxiety",
            "Avoid Notifications",
            "Follow Study Routine",
            "Reduce Screen Time",
            "Study Break Frequency",
            "Max Continuous Reading Time",
            "Unintentional Inattention",
            "Attention Span Decreased"
        ],
        "Significant Finding": [
            "Females had longer attention spans (30–60 min); males clustered around 15–30 min.",
            "0–4 hours of use linked to better attention; 6+ hours linked to shorter spans (2–15 min).",
            "Frequent viewers had lower spans; rare/never watchers had longer spans.",
            "Frequent fatigue linked to shorter spans; rare fatigue linked to better endurance.",
            "More anxiety correlated with shorter attention spans.",
            "Those who avoided distractions had longer attention spans.",
            "Regular routine linked to 60-minute spans; irregular routine linked to shorter spans.",
            "Those who reduced screen time had longer spans.",
            "Breaks every 30–60 min linked to longer spans; breaks every 15 min linked to shorter spans.",
            "Reading ≥30 minutes continuously linked to longer attention spans.",
            "Frequent inattention linked to shorter spans; those without this issue had 30–60 min spans.",
            "Those perceiving a decrease reported lower spans; those not noticing a decline had better maintenance."
        ],
        "Interpretation": [
            "Gender difference in sustained attention.",
            "Excessive use reduces attention.",
            "Short video consumption negatively affects focus.",
            "Mental fatigue lowers attention span.",
            "Anxiety disrupts concentration.",
            "Avoiding notifications improves focus.",
            "Consistent study habits enhance attention.",
            "Screen time reduction helps focus.",
            "Balanced break-taking improves attention.",
            "Reading endurance indicates better attention.",
            "Lack of unintentional distractions improves attention.",
            "Perceived decline correlates with reduced attention."
        ],
        "📉 Symbolic Trend": [
            "Male 🔽 Female 🔼",
            "0–4hr 🔼 6+hr 🔽",
            "Frequent 🔽 Rare 🔼",
            "Frequent 🔽 Rare 🔼",
            "High Anxiety 🔽 Low 🔼",
            "Avoid 🔼 Not Avoid 🔽",
            "Regular 🔼 Irregular 🔽",
            "Reduced 🔼 Not Reduced 🔽",
            "30–60min 🔼 15min 🔽",
            "≥30min 🔼 <30min 🔽",
            "Frequent 🔽 Rare 🔼",
            "Declined 🔽 Not 🔼"
        ]
    }
    trends_df = pd.DataFrame(data)

    # Correlation Analysis
    import pandas as pd

    # Data প্রস্তুত করা
    data = {
        "Sr.": [1, 2, 3, 4, 5, 6],
        "Variables Compared": [
            "Study Hours at Home vs. Average Attention Span",
            "Study Hours at Home vs. Daily Digital Device Usage",
            "Max Continuous Reading Time vs. Attention Span",
            "Attention Span vs. Daily Digital Device Usage",
            "Age vs. Other Variables",
            "Study Break Frequency vs. Attention Span"
        ],
        "Correlation(r)": [
            "+0.36",
            "−0.48",
            "+0.40",
            "−0.27",
            "−0.18 (with study hours), +0.11 (with device use)",
            "+0.15"
        ],
        "Strength": [
            "Moderate (+)",
            "Strong (−)",
            "Moderate (+)",
            "Weak (−)",
            "Weak",
            "Weak (+)"
        ],
        "Interpretation": [
            "More study hours at home are associated with better attention span.",
            "Students who study more at home tend to use digital devices less.",
            "Longer uninterrupted reading is linked to stronger attention span.",
            "Higher digital device usage may lead to reduced attention span.",
            "Age shows weak and inconsistent relationships with other variables.",
            "Taking breaks has a minor positive effect on attention span."
        ]
    }
    corr_df = pd.DataFrame(data)


    # hypothesis Testing & Results
    data = {
        "Hypothesis": [
            "Gender vs. Attention Span",
            "Watching Short Videos vs. Attention Span",
            "Study Routine vs. Attention Span",
            "Daily Digital Device Usage vs. Attention Span",
            "Relationship Status vs. Attention Span",
            "Study Hours at Home vs. Attention Span",
            "Most Attentive Time vs. Attention Span",
            "Pre-Study Anxiety vs. Attention Span",
            "Physical Activity vs. Attention Span",
            "Most Used Platform vs. Attention Span",
            "Device Stress vs. Attention Span",
            "Unintentional Inattention vs. Attention Span",
            "Meditation vs. Attention Span",
            "Digital Device Usage vs. Max Reading Time",
            "Family/Friends Pressure vs. Attention Span",
            "Perceived Change in Attention Span vs. Current Attention Span"
        ],
        "Null Hypothesis (H₀)": [
            "No significant difference in attention span between male and female students",
            "Watching short videos has no significant effect on attention span",
            "Study routine has no significant influence on attention span",
            "No linear correlation between device usage and attention span",
            "Relationship status has no significant effect on attention span",
            "No significant relationship between study hours and attention span",
            "Most attentive time has no significant effect on attention span",
            "Pre-study anxiety has no significant influence on attention span",
            "Physical activity has no significant effect on attention span",
            "Primary platform used has no significant effect on attention span",
            "Device stress does not significantly affect attention span",
            "Unintentional inattention has no significant effect on attention span",
            "Meditation has no significant effect on attention span",
            "Device usage does not significantly impact max reading time",
            "Family/friends pressure has no effect on attention span",
            "Perceived change in attention span has no relation to current attention span"
        ],
        "Alternative Hypothesis (H₁)": [
            "Significant difference in attention span between male and female students",
            "Watching short videos significantly affects attention span",
            "Study routine significantly influences attention span",
            "Significant linear correlation between device usage and attention span",
            "Relationship status significantly affects attention span",
            "Significant relationship between study hours and attention span",
            "Most attentive time significantly affects attention span",
            "Pre-study anxiety significantly influences attention span",
            "Physical activity significantly affects attention span",
            "Platform used significantly influences attention span",
            "Device stress significantly affects attention span",
            "Unintentional inattention significantly affects attention span",
            "Meditation significantly affects attention span",
            "Device usage significantly impacts max reading time",
            "Family/friends pressure significantly affects attention span",
            "Perceived change in attention span significantly relates to current attention span"
        ],
        "Test": [
            "Independent Samples t-test",
            "One-way ANOVA / t-test",
            "Independent Samples t-test",
            "Pearson Correlation Test",
            "One-way ANOVA",
            "Pearson Correlation Test",
            "One-way ANOVA",
            "One-way ANOVA",
            "Independent Samples t-test",
            "One-way ANOVA",
            "One-way ANOVA",
            "One-way ANOVA",
            "Independent Samples t-test",
            "Pearson Correlation Test",
            "One-way ANOVA",
            "ANOVA / Kruskal-Wallis"
        ],
        "p-value": [
            0.312, 0.0001, "< 0.001", "< 0.001", 0.062, "< 0.001",
            0.08, 0.024, 0.017, 0.037, "< 0.001", "< 0.001", 0.025, "< 0.001", 0.001, 0.02
        ],
        "Significant? (p < 0.05)": [
            "No", "Yes", "Yes", "Yes", "No", "Yes",
            "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"
        ],
        "Decision": [
            "❌(Fail to Reject H₀)Fail to Reject H₀", "✅Reject H₀", "✅Reject H₀", "✅Reject H₀", "❌Fail to Reject H₀", "✅Reject H₀",
            "❌Fail to Reject H₀", "✅Reject H₀", "✅Reject H₀", "✅Reject H₀", "✅Reject H₀", "✅Reject H₀",
            "✅Reject H₀", "✅Reject H₀", "✅Reject H₀", "✅Reject H₀"
        ]
    }
    hypothesis_df = pd.DataFrame(data)

    # hypothesis_table = """
    # | Sr. | Hypothesis | Null Hypothesis (H₀) | Alternative Hypothesis (H₁) | Test | p-value | Significant? (p < 0.05) | Decision |
    # |-----|------------|-----------------------|------------------------------|------|---------|--------------------------|-----------|
    # | 1   | Gender vs. Attention Span | No significant difference in attention span between male and female students | Significant difference in attention span between male and female students | Independent Samples t-test | 0.312 | No | Fail to Reject H₀ |
    # | 2   | Watching Short Videos vs. Attention Span | Watching short videos has no significant effect on attention span | Watching short videos significantly affects attention span | One-way ANOVA / t-test | 0.0001 | Yes | Reject H₀ |
    # | 3   | Study Routine vs. Attention Span | Study routine has no significant influence on attention span | Study routine significantly influences attention span | Independent Samples t-test | < 0.001 | Yes | Reject H₀ |
    # | 4   | Daily Digital Device Usage vs. Attention Span | No linear correlation between device usage and attention span | Significant linear correlation between device usage and attention span | Pearson Correlation Test | < 0.001 | Yes | Reject H₀ |
    # | 5   | Relationship Status vs. Attention Span | Relationship status has no significant effect on attention span | Relationship status significantly affects attention span | One-way ANOVA | 0.062 | No | Accept H₀ |
    # | 6   | Study Hours at Home vs. Attention Span | No significant relationship between study hours and attention span | Significant relationship between study hours and attention span | Pearson Correlation Test | < 0.001 | Yes | Reject H₀ |
    # | 7   | Most Attentive Time vs. Attention Span | Most attentive time has no significant effect on attention span | Most attentive time significantly affects attention span | One-way ANOVA | 0.08 | No | Fail to Reject H₀ |
    # | 8   | Pre-Study Anxiety vs. Attention Span | Pre-study anxiety has no significant influence on attention span | Pre-study anxiety significantly influences attention span | One-way ANOVA | 0.024 | Yes | Reject H₀ |
    # | 9   | Physical Activity vs. Attention Span | Physical activity has no significant effect on attention span | Physical activity significantly affects attention span | Independent Samples t-test | 0.017 | Yes | Reject H₀ |
    # | 10  | Most Used Platform vs. Attention Span | Primary platform used has no significant effect on attention span | Platform used significantly influences attention span | One-way ANOVA | 0.037 | Yes | Reject H₀ |
    # | 11  | Device Stress vs. Attention Span | Device stress does not significantly affect attention span | Device stress significantly affects attention span | One-way ANOVA | < 0.001 | Yes | Reject H₀ |
    # | 12  | Unintentional Inattention vs. Attention Span | Unintentional inattention has no significant effect on attention span | Unintentional inattention significantly affects attention span | One-way ANOVA | < 0.001 | Yes | Reject H₀ |
    # | 13  | Meditation vs. Attention Span | Meditation has no significant effect on attention span | Meditation significantly affects attention span | Independent Samples t-test | 0.025 | Yes | Reject H₀ |
    # | 14  | Digital Device Usage vs. Max Reading Time | Device usage does not significantly impact max reading time | Device usage significantly impacts max reading time | Pearson Correlation Test | < 0.001 | Yes | Reject H₀ |
    # | 15  | Family/Friends Pressure vs. Attention Span | Family/friends pressure has no effect on attention span | Family/friends pressure significantly affects attention span | One-way ANOVA | 0.001 | Yes | Reject H₀ |
    # | 16  | Perceived Change in Attention Span vs. Current Attention Span | Perceived change in attention span has no relation to current attention span | Perceived change in attention span significantly relates to current attention span | ANOVA / Kruskal-Wallis | 0.02 | Yes | Reject H₀ |
    # """

    ###########################################################

    # Interpretation of Results in Context of Research Questions (RQ1–RQ8)
    # research_question_table = """
    # | RQ  | Research Question                                                                 | Summary of Insights                                                                                         | Statistical/Model Evidence                                                                                                    | Conclusion                                         |
    # |-----|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
    # | RQ1 | What is the current attention span of college students in urban and suburban areas of Bangladesh during academic activities? | Mean = 28.7 mins, Median = 30 mins; Majority from city (296/349); No major urban vs. suburban differences    | Descriptive stats; Institution location not significant                                                                       | Moderate average span; Location has minimal impact |
    # | RQ2 | How do digital device usage patterns affect students’ attention span?             | High screen time and short video watching linked to lower attention                                          | ANOVA (p = 0.0001); Correlation (p < 0.001); XGBoost: watch_short_videos, use_digital_devices_while_studying among top features | Strong negative impact from digital engagement     |
    # | RQ3 | What physiological and psychological factors influence attention span?            | Mental fatigue, anxiety, poor sleep reduce attention; Meditation & physical activity help                     | p < 0.05 for fatigue, anxiety, sleep; XGBoost: mental_fatigue_frequency, practice_meditation, regular_physical_activity in top 10 | Psychological & lifestyle variables are critical predictors |
    # | RQ4 | Do external distractions impact students’ ability to focus?                       | Daydreaming (56.7%), noise, notifications common distractions                                               | p = 0.048 (family/friend pressure); Moderate model importance                                                                 | Distractions affect focus but are not primary determinants |
    # | RQ5 | Are there demographic differences in attention span?                              | No major role of gender, age, or education level                                                             | Gender (p = 0.312); Uniform education level; Model: demographic features ranked low                                           | Demographics are weak predictors                  |
    # | RQ6 | How effective are self-regulation strategies in improving attention?              | Study routine, breaks, reduced screen time strongly linked to better attention                               | Study routine (p < 0.001); XGBoost: study_hours_home, max_continuous_reading_time, study_break_frequency in top 10             | Self-regulation strategies are highly effective   |
    # | RQ7 | What are main distractions and frequency of inattention?                          | Daydreaming most common; ~91% report some level of unintentional inattention                                 | ANOVA (p < 0.001); High self-reported frequency                                                                               | A major challenge to learning; needs behavioral interventions |
    # | RQ8 | How do students perceive change in attention over time?                           | 56.16% believe their span has decreased in the digital age                                                   | p = 0.0124 for attention_span_decreased; Digital usage supports their claim                                                   | Students’ perception aligns with model & statistical results |
    # """

    data = {
        "RQ": ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6", "RQ7", "RQ8"],
        "Research Question": [
            "What is the current attention span of college students in urban and suburban areas of Bangladesh during academic activities?",
            "How do digital device usage patterns affect students’ attention span?",
            "What physiological and psychological factors influence attention span?",
            "Do external distractions impact students’ ability to focus?",
            "Are there demographic differences in attention span?",
            "How effective are self-regulation strategies in improving attention?",
            "What are main distractions and frequency of inattention?",
            "How do students perceive change in attention over time?",
        ],
        "Summary of Insights": [
            "Mean = 28.7 mins, Median = 30 mins; Majority from city (296/349); No major urban vs. suburban differences",
            "High screen time and short video watching linked to lower attention",
            "Mental fatigue, anxiety, poor sleep reduce attention; Meditation & physical activity help",
            "Daydreaming (56.7%), noise, notifications common distractions",
            "No major role of gender, age, or education level",
            "Study routine, breaks, reduced screen time strongly linked to better attention",
            "Daydreaming most common; ~91% report some level of unintentional inattention",
            "56.16% believe their span has decreased in the digital age",
        ],
        "Statistical/Model Evidence": [
            "Descriptive stats; Institution location not significant",
            "ANOVA (p = 0.0001); Correlation (p < 0.001); XGBoost: watch_short_videos, use_digital_devices_while_studying among top features",
            "p < 0.05 for fatigue, anxiety, sleep; XGBoost: mental_fatigue_frequency, practice_meditation, regular_physical_activity in top 10",
            "p = 0.048 (family/friend pressure); Moderate model importance",
            "Gender (p = 0.312); Uniform education level; Model: demographic features ranked low",
            "Study routine (p < 0.001); XGBoost: study_hours_home, max_continuous_reading_time, study_break_frequency in top 10",
            "ANOVA (p < 0.001); High self-reported frequency",
            "p = 0.0124 for attention_span_decreased; Digital usage supports their claim",
        ],
        "Conclusion": [
            "Moderate average span; Location has minimal impact",
            "Strong negative impact from digital engagement",
            "Psychological & lifestyle variables are critical predictors",
            "Distractions affect focus but are not primary determinants",
            "Demographics are weak predictors of attention",
            "Self-regulation strategies are highly effective",
            "A major challenge to learning; needs behavioral interventions",
            "Students’ perception aligns with model & statistical results",
        ]
    }

    df_rq = pd.DataFrame(data)

    # key insights from EDA
    import pandas as pd

    data = {
        # "No": list(range(1, 13)),
        "Variable": [
            "Average Attention Span",
            "Study Hours at Home",
            "Max Continuous Reading Time",
            "Daily Digital Device Usage",
            "Study Break Frequency",
            "Watching Short Videos",
            "Mental Fatigue Frequency",
            "Digital Device Use While Studying",
            "Most Used Digital Platform",
            "Regular Physical Activity",
            "Attention During Class",
            "Most Attentive Time of Day"
        ],
        "Key Insight / Distribution Summary": [
            "Most common: 30 min (32.4%), Range: 2–60 min",
            "Majority: 1–4 hours, Symmetrical distribution",
            "Common: 15–30 min, Left-skewed distribution",
            "Mostly 2–4 hours/day, few extreme users",
            "Majority take breaks every 15–30 min",
            "41% regular, 41.3% sometimes, 17.8% never",
            "45.1% report 'sometimes' feeling fatigue",
            "48.4% use 'sometimes'",
            "YouTube (33%) > Facebook (30%) > Instagram (25%)",
            "66.7% do not exercise regularly",
            "50.4% report paying attention",
            "Evening (41%) > Morning (29%) > Late Night (28%)"
        ],
        "Interpretation": [
            "Varies widely among students",
            "Balanced study routine",
            "Moderate sustained focus",
            "Controlled usage for most",
            "Frequent short breaks preferred",
            "High exposure to short-form content",
            "Moderate cognitive strain",
            "Devices moderately distract during study",
            "Visual platforms most popular",
            "Low physical activity overall",
            "Mixed classroom engagement",
            "Focus peaks on evening hours"
        ]
    }

    eda_df = pd.DataFrame(data)





    # Streamlit এ দেখানো
    st.subheader("⚙ `Various Test Results Analysis`")
    col1,col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        eda = st.checkbox('EDA insights')
    if eda:
        st.write('### `Exploratory Data Analysis Insights`')
        st.dataframe(eda_df)

    with col2:
        chi = st.checkbox("Chi-square Test Analysis")
    if chi:
        st.write('### `Chi-square Test Analysis Insights`')
        st.dataframe(chi_df)

    with col3:
        trends = st.checkbox('Significant Trends Analysis(p < 0.05)')
    if trends:
        st.write('### `Significant Trends Analysis(p < 0.05) Results`')
        st.dataframe(trends_df)

    with col4:
        corr = st.checkbox('Correlation Coefficients Analysis')
    if corr:
        st.write('### `Correlation Coefficients Analysis Results`')
        st.dataframe(corr_df)

    with col5:
        hypothesis = st.checkbox('Hypothesis Tests Analysis')
    if hypothesis:
        st.write('### `Hypothesis Tests Analysis Results`')
        st.dataframe(hypothesis_df)
        # st.write(hypothesis_table)

    with col6:
        research = st.checkbox('Research Questions Results')
    if research:
        st.write('### `Research Questions Results`')
        st.dataframe(df_rq)
        # st.write(research_question_table)
    st.write('---')

    #################################################################
    # XGBoost Model Evaluation

    # classi_mark_report markdown
    classi_mark_report = '''
    | Class        | Precision | Recall | F1-Score | Support |
    |--------------|-----------|--------|----------|---------|
    | **High**     | 0.84      | 0.84   | 0.84     | 37      |
    | **Low**      | 0.91      | 0.84   | 0.87     | 37      |
    | **Medium**   | 0.70      | 0.76   | 0.73     | 37      |
    | **Accuracy** |           |        | **0.81** | 111     |
    | **Macro Avg**| 0.82      | 0.81   | 0.73     |         |
    | **Weighted Avg** | 0.82  | 0.81   | 0.81     |         |
    '''

    # confusion_matrix_report markdown
    confusion_matrix_report = '''
    |          | High | Low | Medium |
    |:--------:|:----:|:---:|:------:|
    | High     | 31   | 3   | 3      |
    | Low      | 2    | 31  | 4      |
    | Medium   | 4    | 5   | 28     |
    '''

    # feature_importance_report markdown
    feature_importance_report = '''
    `Top 10 Important Features (with Bar Representation)`

    | Rank | Feature                           | Importance | Visual Scale (0 - 0.12) |
    |------|-----------------------------------|------------|--------------|
    | 1    | max_continuous_reading_time       | 0.120106   | ██████████ 0.12 |
    | 2    | study_hours_home                  | 0.087309   | ████████ 0.09 |
    | 3    | practice_meditation               | 0.056207   | █████ 0.06 |
    | 4    | relationship_status               | 0.044573   | ████ 0.04 |
    | 5    | watch_short_videos                | 0.042300   | ████ 0.04 |
    | 6    | difficulty_focusing_online_class  | 0.039540   | ████ 0.04 |
    | 7    | use_digital_devices_while_studying| 0.037665   | ████ 0.04 |
    | 8    | regular_physical_activity         | 0.037253   | ████ 0.04 |
    | 9    | attention_in_class                | 0.037005   | ████ 0.04 |
    | 10   | mental_fatigue_frequency          | 0.034038   | ███ 0.03 |
    '''


    st.subheader('💣 `XGBoost Model Evaluation`')
    col1, col2, col3 = st.columns(3)
    with col1:
        classification_report = st.checkbox('Classification Report')
    if classification_report:
        st.write('### 📝`Classification Report of XGBoosT Classifier`')
        st.write(classi_mark_report)
        interpretation = '''
                    - **`The XGBoost classifier` achieved a `strong accuracy of 81.08%` on the `test set`.** 
                    - **with `F1-Scores` ranging from `0.73 to 0.87` across the `three classes`.** 
                    - **The `Low Attention group` had the `highest performance` with `91% precision` and `84% recall`.**
                    - **The `High Attention group` was also `well-classified`, with `84% precision` and `recall`.**
                    - **The `Medium Attention group`, though `slightly lower` in `precision (0.70)`, had solid `recall (0.76)`,** 
                    - **indicating that the `model captured the majority of medium-range students accurately`.**
                    - **The classifier demonstrated `particularly high performance in identifying both High and Low attention groups`.**
    '''
        img = Image.open("imgs/model/classification_report.png")  
        st.image(img)
        st.write(interpretation)

    # with col2:
    #     classification_report_figure = st.checkbox('Classification Report-Figure')
    # if classification_report_figure:
    #     img = Image.open("imgs/model/classification_report.png")  
    #     st.image(img)

    with col2:
        confusion_matrix = st.checkbox('Confusion Matrix')
    if confusion_matrix:
        st.write('#### 📝 `Confusion Matrix Report of XGBoosT Classifier`')
        st.write(confusion_matrix_report)
        img = Image.open('imgs/model/confusion_matrix.png')
        st.image(img)
        interpetation = '''
                        - **`High Attention Class`**: **`84%`** students (**`31`** `correctly classified` out of **`37`**)  
                        - **`Low Attention Class`**: **`84%`** students (**`31`** `correctly classified` out of **`37`**)  
                        - **`Medium Attention Class`**: **`76%`** students (**`28`** `correctly classified` out of **`37`**)  
    '''
        st.write(interpetation)

    with col3:
        feature_importance = st.checkbox('Feature Importance Report')
    if feature_importance:
        st.write('### 📝`Feature Importance Report of XGBoosT Classifier`')
        st.write(feature_importance_report)
        # img = Image.open("imgs/model/feature_importance.png")  
        # st.image(img)
        interpretation = '''
                            The Model **`confirms clear association`** between
                            - **`Study discipline`**
                            - **`Psychological trats`**
                            - **`Digital habits`** &
                            - **`attention span`**
                            While `demographic factors` plays a **`negligible`** role.
                        '''
        st.write(interpretation)
    st.write('---')


    ###################################################################
    # figures of the project
    st.subheader("📸 `Project Figure Gallery`")
    photos = st.checkbox('`Show Figures`')
    if photos:
        # folder path
        image_folder = "imgs/figures"

        # all images load from folder
        images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]

        # showing figures into 3 columns
        cols = st.columns(3)
        for i, img in enumerate(images):
            with cols[i % 3]:
                st.image(img)







#################################################################################
#################################################################################
# --- Page 3: Prediction ---
elif page == "😎 Prediction":
    import pandas as pd
    import streamlit as st
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import xgboost as xgb
    from PIL import Image

    # ------------------------------
    # Cached training pipeline
    # ------------------------------
    @st.cache_resource
    def load_resources():
        df = load_data()

        # Target
        def categorize_attention(score):
            if score <= 10: return 'Low'
            elif score <= 30: return 'Medium'
            else: return 'High'
        df['attention_category'] = df['average_attention_span'].apply(categorize_attention)

        target_col = 'attention_category'
        X = df.drop(columns=[target_col, 'average_attention_span'])
        y = df[target_col]

        # Encode categorical features
        categorical_columns = [col for col in X.columns if X[col].dtype == 'object']
        encoders = {}
        for col in categorical_columns:
            le_col = LabelEncoder()
            X[col] = le_col.fit_transform(X[col].astype(str))
            encoders[col] = le_col

        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # SMOTE (all features)
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y_encoded)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
        )

        # Train model (all features)
        model_full = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
        model_full.fit(X_train, y_train)

        # Accuracy with all features
        y_pred_full = model_full.predict(X_test)
        acc_full = accuracy_score(y_test, y_pred_full)

        # Feature importance
        importances = model_full.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        top_features = feature_importance_df['feature'].head(10).tolist()

        # Retrain on top 10 features
        X_top = X[top_features]
        X_top_scaled = scaler.fit_transform(X_top)
        X_res_top, y_res_top = smote.fit_resample(X_top_scaled, y_encoded)

        X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
            X_res_top, y_res_top, test_size=0.2, stratify=y_res_top, random_state=42
        )

        model_top = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
        model_top.fit(X_train_top, y_train_top)

        # Accuracy with top 10 features
        y_pred_top = model_top.predict(X_test_top)
        acc_top = accuracy_score(y_test_top, y_pred_top)

        return df, X, le, scaler, encoders, model_top, top_features, acc_full, acc_top, feature_importance_df

    df, X, le, scaler, encoders, model_top, top_features, acc_full, acc_top, feature_importance_df = load_resources()

    # ------------------------------
    # Streamlit UI
    # ------------------------------
    st.title("`Prediction: Attention Category`")
    img = Image.open("imgs/predic.png")
    st.image(img)
    st.subheader('`Attention Span Classifier - XGBoost Model`')

    col1, col2 = st.columns(2)
    with col1:
        classification_report_chk = st.checkbox('Show Classification Report')
    with col2:
        recommendation_chk = st.checkbox('Recommendation & Implication')

    classification_report_top10 = '''
    #### 📊 Classification Report (Top 10 features)
    | Class   | Precision | Recall | F1-Score | Support |
    |---------|-----------|--------|----------|---------|
    | High    | 0.82      | 0.76   | 0.79     | 37      |
    | Low     | 0.63      | 0.78   | 0.70     | 37      |
    | Medium  | 0.68      | 0.57   | 0.62     | 37      |
    | **Accuracy** |       |        | 0.70     | 111     |
    | **Macro Avg**| 0.71  | 0.70  | 0.70     | 111     |
    | **Weighted Avg**| 0.71 | 0.70 | 0.70    | 111     |
    '''
    recommendation_implication = '''
    #### ✅ Recommendation & Implication for students & institutions
    | 🎓 Students                                           | 🏫 Educators & Institutions               | 🏛️ Policy Makers                                       |
    | ----------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------- |
    | Prioritize sustained reading tasks                    | Introduce digital mindfulness programs    | National guidelines on screen time for students         |
    | Adopt structured study routines and breaks            | Redesign online learning platforms        | Infrastructure investment for healthy digital education |
    | Limit short video consumption                         | Encourage interactive teaching methods    | Training for educators on attention-aware teaching      |
    | Engage in regular physical activity & rest            | Provide mental health support             | Monitor and research digital health                     |
    | Practice digital mindfulness and reduce notifications | Build digital-free zones or periods       | Public awareness campaigns                              |
    |                                                       | Use predictive insights for early support |                                                         |
    '''

    if classification_report_chk: st.write(classification_report_top10)
    if recommendation_chk: st.write(recommendation_implication)

 

    # ------------------------------
    # Sidebar user input
    # ------------------------------
    def user_report(top_features, df):
        user_report_data = {}
        for feature in top_features:
            if df[feature].dtype in ['int64', 'float64']:
                if feature == 'max_continuous_reading_time':
                    val = st.sidebar.slider(f"{feature}", 0.0, 50.0, 30.0)
                else:
                    min_val, max_val, mean_val = float(df[feature].min()), float(df[feature].max()), float(df[feature].mean())
                    val = st.sidebar.slider(f"{feature}", min_val, max_val, mean_val)
            else:
                options = df[feature].dropna().unique().tolist()
                val = st.sidebar.selectbox(f"{feature}", options)
            user_report_data[feature] = val
        return pd.DataFrame(user_report_data, index=[0])
  

    show_input = st.sidebar.checkbox('Show Prediction Options:')
    if show_input:
        '---'
        user_data = user_report(top_features, df)
        st.subheader('`Predicted Attention Span Category`')
        st.write("`User Input for Top 10 Features`")
        st.write(user_data)

        # Real-time Prediction
        with st.spinner('Predicting...'):
            for col in user_data.columns:
                if col in X.columns and df[col].dtype == 'object':
                    user_data[col] = encoders[col].transform(user_data[col].astype(str))
            input_scaled = scaler.transform(user_data)
            pred_encoded = model_top.predict(input_scaled)
            pred_class = le.inverse_transform(pred_encoded)[0]
            pred_proba = model_top.predict_proba(input_scaled)[0][pred_encoded[0]] * 100
            st.subheader(f" **`Predicted Category`: `{pred_class}` (`{pred_proba:.2f}% probability`)**")

    

       # ------------------------------
    # Extra model results (কমেন্ট করে রাখা হলো)
    # ------------------------------
    st.write('---')
    st.subheader("🔎 `Model Accuracy Comparison`")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**`Accuracy with ALL ({X.shape[1]} features)`:** `{acc_full*100:.2f}%`")
    with col2:
        st.write(f"**`Accuracy with TOP 10 features`:** `{acc_top*100:.2f}%`")
    with col3:
        top10_imp = st.checkbox('`Feature Importance(Top 10)`')
    if top10_imp:
        st.dataframe(feature_importance_df.head(10))








