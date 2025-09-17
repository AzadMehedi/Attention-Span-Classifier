
# 🧾 `Attention Span Classifier`

<!-- ### **An Assessment of Attention Span among College Students in Dhaka: Patterns, Potential Implications, and Influencing Factors.** -->
### <span style="color:green;">An Assessment of Attention Span among College Students in Dhaka: Patterns, Potential Implications, and Influencing Factors.</span>


### `Live` : `https://attention-span-classifier.streamlit.app/`

---

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a hred="#bacground">Bakground of The Research</a> 
- <a href="#objectives">Research Objectives</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#data-cleaning--preparation">Data Cleaning & Preparation</a>
- <a href="#eda">Exploratory Data Analysis (EDA)</a>
- <a href="#chisquare-tests">Chi-Square Tests</a>
- <a href="#trend">Trend Analysis</a>
- <a href="#hypothesis-tests">Hypothesis Tests</a>
- <a href="#correlation">Correlation Coefficient  Results</a>
- <a href="#research-questions--key-findings">Research Questions & Key Findings</a>
- <a href="#model">XGBoost Classification Model</a>
- <a href="#model-evaluation">Model Performance Evaluation</a>
- <a href="#confusion-matric">Confusion Matrics</a>
- <a href="#influensing-factor">Top Influencing Factors</a>
- <a href="#recommendations">Recommendations & Implicationss</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>

- <a href="#author--contact">Author & Contact</a>

---
<h2><a class="anchor" id="overview"></a>Overview</h2>

Understanding the `attention span` of students in the digital age holds considerable importance, particularly for educational institutions and stakeholders aiming to enhance academic outcomes and mental well-being. As attention is one of the foundational elements of learning, any shift or disruption in students’ ability to maintain focus may directly influence their academic performance, classroom behavior, and overall engagement with learning materials. So, this research will deep dive into students attention span related analysises & classify students into `HIGH`, `MEDIUM` & `LOW` attentiion categories using `XGBoost Classifier` machine learning model.

---
<h2><a class="anchor" id="bacground"></a>Bakground of The Research</h2>

- Students Attention span plays a crucial role in academic performance that lead a successful career & shines in personal life. 
- Attention span of students are shrinking, both locally & globally.
- Digital devices dependency, bad influence of social media, notification, reels plays a vital role to decrease of attention span.
- Dhaka as the capital city has a high density of college students with diverse background.



<h2><a class="anchor" id="objectives"></a>Research Objectives</h2>

`Research Objectives are:`
1.	To assess the attention span among college students of Dhaka.
2.	To identify the patterns & influencing factors of attention span.
3.	To classify students based on their attention category (High, Medium, Low)
4.	Suggesting potential implications on academic performance. 



---
<h2><a class="anchor" id="dataset"></a>Dataset</h2>

- Single CSV files located in `https://github.com/AzadMehedi/Attention-Span-Classifier/` folder.
- **Source** : Data collected through both structured `printed questionnaire` & `google form`.
- **Sampling Technique** : N`on-probability convenience sampling technique`
- **Dataset Details** : 
   - 349 rows
   - 31 columns

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

- Python (Pandas, Numpy, Matplotlib, Seaborn, PLotly, Scikit Learn)
- Statistics (Statistical Tests, Hypothesis Tests etc.)
- Streamlit (Web Application & hosting)
- GitHub
- VScode, Colab Notebook, Jupyter Notebook

---
<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
Attention-Span_Classifier/
│
├── imgs
|  ├── figires
│  ├── model
├── LICENSE
├── README.md
├── cleaned_dataset
├── final.py
├── model.sav
├── requirements.txt

```

---
<h2><a class="anchor" id="data-cleaning--preparation"></a>Data Cleaning & Preparation</h2>

- Remove Missing values
- Remove Duplicate values
- Created summary tables with vendor-level metrics
- Data Type Conversion:
   - Numeric variables such as age, daily device usage (in hours), and attention span (in minutes) were converted to integer types. 
   - Categorical variables were converted to strings or categorical types for easier grouping and comparison.
- Outlier Detection & treatment
- Variable Renaming and Labeling 

---
<h2><a class="anchor" id="eda"></a>Exploratory Data Analysis (EDA)</h2>

### **`Descriptive Analysis of Categorical Variables: `**

- 🎥 **watch_short_videos:** About **144 students** watch short videos "Sometimes," making it the most common response.  
- 📱 **use_digital_devices_while_studying:** Around **169 students** use digital devices "Sometimes" while studying.  
- 😟 **pre_study_anxiety:** Nearly **122 students** experience pre-study anxiety "Sometimes."  
- 😴 **mental_fatigue_frequency:** About **154 students** report feeling mental fatigue "Sometimes."  
- 🌥 **biggest_study_distraction:** The top distraction is **Daydreams (198 students)**, higher than any other distraction.  
- 🛌 **regular_sufficient_sleep:** Around **165 students** get regular sufficient sleep.  



### **`Descriptive Summary of Numerical Variables:`**

- **Study Hours at Home**  
  - Average: **4.36 hours**, Median: **4 hours**  
  - Moderate variation (**Std. Dev = 1.84**), Range: **1–7 hours**  
  - Suggests a fairly balanced study routine at home

- **Average Attention Span**  
  - Mean: **28.7 minutes**, Median: **30 minutes**  
  - High variability (**Std. Dev = 19.42**), Range: **2–60 minutes**  
  - Indicates attention span varies widely among students

- **Study Break Frequency**  
  - Average break frequency: **27.5 minutes**, Median: **30 minutes**  
  - Large variation (**Std. Dev = 21.58**), Range: **0–60 minutes**  
  - Shows students prefer frequent short breaks, but habits differ

- **Daily Digital Device Usage**  
  - Average usage: **3.07 hours/day**, Median: **2 hours/day**  
  - Wide range (**0–8 hours**) indicates uneven device usage patterns  
  - Most students use devices moderately

- **Max Continuous Reading Time**  
  - Average: **22.2 minutes**, Median: **30 minutes**  
  - Lower variation (**Std. Dev = 9.45**), Range: **3–30 minutes**  
  - Suggests students can maintain moderate sustained focus



**The key insights derived from EDA:**

- ⏱ **Average Attention Span:** Most students concentrate for about **30 minutes**, but range is wide (2–60 mins).  
- 📚 **Study Hours at Home:** Majority study **1–4 hours daily**, showing a **balanced study pattern**.  
- 📖 **Max Continuous Reading Time:** Commonly **15–30 mins**, indicating **moderate sustained focus**.  
- 📱 **Daily Digital Device Usage:** Mostly **2–4 hours/day**, with few heavy users.  
- ☕ **Study Break Frequency:** Students prefer **short, frequent breaks (15–30 mins)**.  
- 🎥 **Watching Short Videos:** High exposure — **83% watch regularly or sometimes**.  
- 😴 **Mental Fatigue:** Around **45% report occasional fatigue**, suggesting **moderate strain**.  
- 📵 **Device Use While Studying:** Nearly **half get distracted sometimes** by devices.  
- ▶️ **Most Used Platform:** **YouTube > Facebook > Instagram** — visual media dominates.  
- 🏃 **Regular Physical Activity:** **Two-thirds lack regular exercise**, indicating low activity.  
- 🏫 **Attention During Class:** **Half the students pay attention**, others struggle.  
- 🌆 **Most Attentive Time:** **Evening is peak focus time (41%)**, followed by morning and late night.  



---
<h2><a class="anchor" id="chisquare-tests"></a>Chi Square Test results</h2>

### `Chi-Square Test Results`

| Variable Name                   | Categories                                      | Chi-square Statistic | p-value  | Significant Association? |
|---------------------------------|------------------------------------------------|--------------------|----------|-------------------------|
| gender                          | Female, Male, Other                            | 23.27              | 0.0098   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| institution_location            | City, Suburban, Village (Rural)               | 7.05               | 0.7208   | <span style="color:red">❌ No (Fail to Reject H₀)</span> |
| age                             | 16, 17, 18, 19, 20                             | 17.31              | 0.6329   | <span style="color:red">❌ No (Fail to Reject H₀)</span> |
| daily_digital_device_usage      | 0, 2, 4, 6, 8                                  | 118.73             | 0.0000   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| most_used_digital_platform      | Facebook, Instagram, Others, TV, TikTok, Twitter, YouTube | 51.78 | 0.0080   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| watch_short_videos              | No, Sometimes, Yes                              | 36.08              | 0.0001   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| have_social_media_account       | Yes                                            | 0.00               | 1.0000   | <span style="color:red">❌ No (Fail to Reject H₀)</span> |
| mental_fatigue_frequency        | Never, Often, Rarely, Sometimes               | 41.71              | 0.0002   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| pre_study_anxiety               | Never, Often, Rarely, Sometimes               | 34.99              | 0.0025   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| regular_sufficient_sleep        | No, Sometimes, Yes                             | 7.72               | 0.6558   | <span style="color:red">❌ No (Fail to Reject H₀)</span> |
| digital_device_stress           | No, Not sure, Yes                               | 10.96              | 0.3607   | <span style="color:red">❌ No (Fail to Reject H₀)</span> |
| biggest_study_distraction       | Daydreams, Noise, Notifications, Others       | 21.32              | 0.1268   | <span style="color:red">❌ No (Fail to Reject H₀)</span> |
| avoid_notifications             | No, Sometimes, Yes                             | 28.06              | 0.0018   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| family_friends_pressure_effect  | No, Sometimes, Yes                             | 10.54              | 0.3949   | <span style="color:red">❌ No (Fail to Reject H₀)</span> |
| follow_study_routine            | No, Yes                                        | 11.82              | 0.0373   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| tried_reduce_screen_time        | No, Sometimes, Yes                             | 28.20              | 0.0017   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| practice_meditation             | No, Sometimes, Yes                             | 8.06               | 0.6230   | <span style="color:red">❌ No (Fail to Reject H₀)</span> |
| study_break_frequency           | 0, 15, 30, 60                                  | 68.80              | 0.0000   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| max_continuous_reading_time     | 3, 10, 15, 30                                  | 110.68             | 0.0000   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| unintentional_inattention       | No, Sometimes, Yes                             | 20.48              | 0.0251   | <span style="color:green">✅ Yes (Reject H₀)</span> |
| attention_span_decreased        | No, Not sure, Yes                              | 18.35              | 0.0493   | <span style="color:green">✅ Yes (Reject H₀)</span> |

---


<h2><a class="anchor" id="trend"></a>Trend Analysis</h2>

### `statistically significant associations with average attention span based on Chi-square tests:`

| Variable                    | Observation / Interpretation                                   | Trend |
|------------------------------|---------------------------------------------------------------|-------|
| Gender                       | Females had longer attention spans (30–60 min); males clustered around 15–30 min. | Male <span style="color:red">🔽</span> Female <span style="color:green">🔼</span> |
| Daily Digital Device Usage   | 0–4 hours linked to better attention; 6+ hours linked to shorter spans. | 0–4hr <span style="color:green">🔼</span> 6+hr <span style="color:red">🔽</span> |
| Watch Short Videos           | Frequent viewers had lower spans; rare/never watchers had longer spans. | Frequent <span style="color:red">🔽</span> Rare <span style="color:green">🔼</span> |
| Mental Fatigue               | Frequent fatigue linked to shorter spans; rare fatigue linked to better endurance. | Frequent <span style="color:red">🔽</span> Rare <span style="color:green">🔼</span> |
| Pre-study Anxiety            | More anxiety correlated with shorter attention spans.        | High Anxiety <span style="color:red">🔽</span> Low <span style="color:green">🔼</span> |
| Avoid Notifications          | Those who avoided distractions had longer attention spans.  | Avoid <span style="color:green">🔼</span> Not Avoid <span style="color:red">🔽</span> |
| Follow Study Routine         | Regular routine linked to 60-minute spans; irregular shorter. | Regular <span style="color:green">🔼</span> Irregular <span style="color:red">🔽</span> |
| Reduce Screen Time           | Screen time reduction helps focus.                           | Reduced <span style="color:green">🔼</span> Not Reduced <span style="color:red">🔽</span> |
| Study Break Frequency        | Breaks every 30–60 min linked to longer spans; 15 min shorter. | 30–60min <span style="color:green">🔼</span> 15min <span style="color:red">🔽</span> |
| Max Continuous Reading Time  | Reading ≥30 min continuously linked to longer attention spans. | ≥30min <span style="color:green">🔼</span> <30min <span style="color:red">🔽</span> |
| Unintentional Inattention    | Frequent inattention linked to shorter spans; rare attention better. | Frequent <span style="color:red">🔽</span> Rare <span style="color:green">🔼</span> |
| Attention Span Decreased     | Those perceiving a decrease reported lower spans; others maintained. | Declined <span style="color:red">🔽</span> Not <span style="color:green">🔼</span> |


---
<h2><a class="anchor" id="hypothesis-tests"></a>Hypothesis Tests</h2>

### 🧪 `Hypothesis Test Results` 

- **Hypothesis 1:** Gender → Attention Span  
    H₀: No significant difference between male and female students  
    H₁: Significant difference exists  
    `Test: t-test, p = 0.312 | Significant? ❌ No | Decision: ❌ Fail to Reject H₀`

- **Hypothesis 2:** Watching Short Videos → Attention Span  
    H₀: Watching short videos has no effect  
    H₁: Watching short videos significantly affects attention  
    `Test: ANOVA, p = 0.0001 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 3:** Study Routine → Attention Span  
    H₀: Following a study routine has no influence  
    H₁: Following a study routine significantly affects attention  
    `Test: t-test, p < 0.001 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 4:** Daily Digital Device Usage → Attention Span  
    H₀: No linear correlation  
    H₁: Significant linear correlation exists  
    `Test: Pearson Correlation, p < 0.001 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 5:** Relationship Status → Attention Span  
    H₀: No effect  
    H₁: Significant effect exists  
    `Test: ANOVA, p = 0.062 | Significant? ❌ No | Decision: ❌ Failed to Reject H0 H₀`

- **Hypothesis 6:** Study Hours at Home → Attention Span  
    H₀: No significant relationship  
    H₁: Significant relationship exists  
    `Test: Pearson Correlation, p < 0.001 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 7:** Most Attentive Time → Attention Span  
    H₀: No effect  
    H₁: Significant effect exists  
    `Test: ANOVA, p = 0.08 | Significant? ❌ No | Decision: ❌ Fail to Reject H₀`

- **Hypothesis 8:** Pre-study Anxiety → Attention Span  
    H₀: No influence  
    H₁: Significant influence exists  
    `Test: ANOVA, p = 0.024 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 9:** Physical Activity → Attention Span  
    H₀: No effect  
    H₁: Significant effect exists  
    `Test: t-test, p = 0.017 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 10:** Most Used Platform → Attention Span  
    H₀: Platform has no effect  
    H₁: Platform significantly affects attention  
    `Test: ANOVA, p = 0.037 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 11:** Device Stress → Attention Span  
    H₀: No effect  
    H₁: Significant effect exists  
    `Test: ANOVA, p < 0.001 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 12:** Unintentional Inattention → Attention Span  
    H₀: No effect  
    H₁: Significant effect exists  
    `Test: ANOVA, p < 0.001 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 13:** Meditation → Attention Span  
    H₀: No effect  
    H₁: Significant effect exists  
    `Test: t-test, p = 0.029 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 14:** Daily Digital Device Usage → Max Reading Time  
    H₀: No effect  
    H₁: Significant relationship exists  
    `Test: Pearson Correlation, p < 0.001 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 15:** Family/Peer Pressure → Attention Span  
    H₀: No effect  
    H₁: Significant effect exists  
    `Test: ANOVA, p = 0.048 | Significant? ✅ Yes | Decision: ✅ Reject H₀`

- **Hypothesis 16:** Perceived Change in Attention Span → Attention Span  
    H₀: No significant relation  
    H₁: Significant relation exists  
    `Test: ANOVA, p = 0.0124 | Significant? ✅ Yes | Decision: ✅ Reject H₀`


---


<h2><a class="anchor" id="correlation"></a>Correlation Coefficient Results</h2>
# Correlation Summary

| Sr. | Variables Compared                           | Correlation (r) | Strength       | Interpretation                                         |
|-----|---------------------------------------------|----------------|----------------|-------------------------------------------------------|
| 1   | Study Hours at Home vs. Average Attention Span | 🟢 +0.36       | Moderate (+)   | More study hours at home are associated with better attention span. |
| 2   | Study Hours at Home vs. Daily Digital Device Usage | 🔴 −0.48       | Strong (−)     | Students who study more at home tend to use digital devices less. |
| 3   | Max Continuous Reading Time vs. Attention Span | 🟢 +0.40       | Moderate (+)   | Longer uninterrupted reading is linked to stronger attention span. |
| 4   | Attention Span vs. Daily Digital Device Usage | 🔴 −0.27       | Weak (−)       | Higher digital device usage may lead to reduced attention span. |
| 5   | Age vs. Other Variables                        | 🔴 −0.18 (with study hours), 🟢 +0.11 (with device use) | Weak           | Age shows weak and inconsistent relationships with other variables. |
| 6   | Study Break Frequency vs. Attention Span      | 🟢 +0.15       | Weak (+)       | Taking breaks has a minor positive effect on attention span. |

---
<h2><a class="anchor" id="research-questions--key-findings"></a>Research Questions & Key Findings</h2>

### `Research Questions – Key Insights`

- **RQ1: Current attention span of college students in urban and suburban areas**  
  - **Summary:** Mean = 28.7 mins, Median = 30 mins; City students = 296/349  
  - **Statistical Evidence:** Descriptive statistics; location not significant  
  - **Conclusion:** Moderate average span; location has little effect  

- **RQ2: Effect of digital device usage patterns on attention span**  
  - **Summary:** High device usage → lower attention span; short video viewers had lower attention  
  - **Statistical Evidence:** p < 0.001 (device usage); ANOVA for short videos (p = 0.0001); feature importance high  
  - **Conclusion:** Strong negative impact from screen time and short videos  

- **RQ3: Physiological and psychological factors influencing attention span**  
  - **Summary:** Mental fatigue, anxiety, and poor sleep linked to low attention  
  - **Statistical Evidence:** p-values < 0.05 for fatigue, sleep, anxiety; top 10 features in model  
  - **Conclusion:** Critical factors for students’ attention regulation  

- **RQ4: Impact of external distractions on students’ focus**  
  - **Summary:** Daydreaming (56.7%), noise, and notifications were top distractions  
  - **Statistical Evidence:** p = 0.048 (peer/family pressure); moderate model importance  
  - **Conclusion:** Environmental distractions matter but are secondary  

- **RQ5: Demographic differences in attention span**  
  - **Summary:** No significant effect of gender, age, or institution type  
  - **Statistical Evidence:** Gender (p = 0.312), age skewed but low impact; education level almost uniform  
  - **Conclusion:** Demographics are not key predictors  

- **RQ6: Effectiveness of self-regulation strategies in improving attention**  
  - **Summary:** Study breaks, routines, screen-time reduction all effective  
  - **Statistical Evidence:** Study routine (p < 0.001); breaks & hours in top features  
  - **Conclusion:** Highly effective; behavioral strategies improve attention  

- **RQ7: Main distractions and frequency of inattention**  
  - **Summary:** Daydreaming most common; 91% report unintentional inattention  
  - **Statistical Evidence:** ANOVA for inattention (p < 0.001); high frequency self-report  
  - **Conclusion:** Major factor affecting learning; deserves intervention  

- **RQ8: Students’ perception of change in attention over time**  
  - **Summary:** 56.16% said their span has decreased in the digital age  
  - **Statistical Evidence:** attention_span_decreased (p = 0.0124); supported by digital usage impact  
  - **Conclusion:** Students' perception matches data trends  


---
<h2><a class="anchor" id="model"></a>Model Building</h2>

### `Model Selection`: `XGBost Classifier`

### `Target variable`
A new column `“attention_category”` was created by categorizing the numeric variable `“average_attention_span”` into three groups.
   - `High`: `(score>30 min)`
   - `Medium`: `(score 11-30 min)`
   - `Low`: `(score<=10 min)`

This setup `excluded` the direct use of `average_attention_span` as a `predictor`, ensuring a more reliable and unbiased classification outcome.

### `Feature Engineering`
- `Label Encoding` (for converting Categorical data into Numerical)
- `Standard Scaler` (for Scaling Numeric data)
- `SMOTE` (for handing class imbalance)

### `Model Training:` 
#### The dataset was split using `stratified sampling` to maintain class balance in both training & testing set.
- **`Training set`**: 80% of the data
- **`Testing set`:** 20% of the data

---

<h2><a class="anchor" id="model-evaluation"></a>Model Performance Evaluation</h2>

- `The XGBoost classifier achieved a strong accuracy of 81.08% on the test set, with F1-Scores ranging from 0.73 to 0.87 across the three classes. `
- The classifier demonstrated particularly `high performance` in identifying both `High and Low`
 attention groups.

![alt text](image.png)

---


<h2><a class="anchor" id="confusion-matric"></a>Confusion Matrics</h2>

The Model correctly classified:
- `84%` of `High attention class` (31 out of 37)
- `84%` of `Low attention class` (31 out of 37)
- `76%` of `Medium attention class` (28 out of 37)

![alt text](image-3.png)

---

<h2><a class="anchor" id="influensing-factor"></a>Influensing Factors</h2>

### **`Feature Importance Ranking`**
<!-- ![alt text](image-4.png) -->

# Feature Importance Ranking

<table>
  <tr>
    <th>Rank</th>
    <th>Feature</th>
    <th>Importance</th>
    <th>Visual Scale (0 - 0.12)</th>
  </tr>
  <tr>
    <td>1</td>
    <td>max_continuous_reading_time</td>
    <td>0.1201</td>
    <td>
      <span style="background-color:#FF4C4C; display:inline-block; width:100%; height:15px;"></span>
    </td>
  </tr>
  <tr>
    <td>2</td>
    <td>Study_hours_home</td>
    <td>0.0873</td>
    <td>
      <span style="background-color:#4C9AFF; display:inline-block; width:73%; height:15px;"></span>
    </td>
  </tr>
  <tr>
    <td>3</td>
    <td>practice_meditation</td>
    <td>0.0562</td>
    <td>
      <span style="background-color:#4CFF88; display:inline-block; width:47%; height:15px;"></span>
    </td>
  </tr>
  <tr>
    <td>4</td>
    <td>relationship_status</td>
    <td>0.0445</td>
    <td>
      <span style="background-color:#FFD24C; display:inline-block; width:37%; height:15px;"></span>
    </td>
  </tr>
  <tr>
    <td>5</td>
    <td>Watch_short_videos</td>
    <td>0.0423</td>
    <td>
      <span style="background-color:#FF8C4C; display:inline-block; width:35%; height:15px;"></span>
    </td>
  </tr>
  <tr>
    <td>6</td>
    <td>dificulty_focusing_online_class</td>
    <td>0.0395</td>
    <td>
      <span style="background-color:#A64CFF; display:inline-block; width:33%; height:15px;"></span>
    </td>
  </tr>
  <tr>
    <td>7</td>
    <td>use_digital_devices_while_studying</td>
    <td>0.0370</td>
    <td>
      <span style="background-color:#4CFFA5; display:inline-block; width:31%; height:15px;"></span>
    </td>
  </tr>
  <tr>
    <td>8</td>
    <td>regular_physical_activity</td>
    <td>0.0372</td>
    <td>
      <span style="background-color:#FF4CA3; display:inline-block; width:31%; height:15px;"></span>
    </td>
  </tr>
  <tr>
    <td>9</td>
    <td>attention_in_class</td>
    <td>0.0370</td>
    <td>
      <span style="background-color:#4CFFEC; display:inline-block; width:31%; height:15px;"></span>
    </td>
  </tr>
  <tr>
    <td>10</td>
    <td>mental_fatigue_frequency</td>
    <td>0.0340</td>
    <td>
      <span style="background-color:#FFB84C; display:inline-block; width:28%; height:15px;"></span>
    </td>
  </tr>
</table>


### The Model confirms clear association between

   - `Study discipline`
   - `Psychological trats`
   - `Digital habits` &
   - `attention span`
- While `demographic factors` plays a `negligible role`.


---


---
<h2><a class="anchor" id="recommendations"></a> Recommendations & Implications</h2>

| For Students                                   | For Educators & Institutions                          | For Policy Makers                                      |
|-----------------------------------------------|------------------------------------------------------|-------------------------------------------------------|
| Prioritize Sustained Reading Tasks            | Introduce Digital Mindfulness Programs              | National Guidelines on Screen Time for Students      |
| Adopt Structured Study Routines and Breaks   | Redesign Online Learning Platforms                  | Infrastructure Investment for Healthy Digital Education |
| Limit Short Video Consumption                 | Encourage Interactive Teaching Methods              | Training for Educators on Attention-Aware Teaching   |
| Engage in Regular Physical Activity & Rest   | Provide Mental Health Support                        | Monitor and Research Digital Health                  |
| Practice Digital Mindfulness and Reduce Notifications | Build Digital-Free Zones or Periods            | Public Awareness Campaigns                            |
|                                               | Use Predictive Insights for Early Support           |                                                       |

---
<h2><a class="anchor" id="how-to-run-this-project"></a>Live This Project</h2>

### `https://attention-span-classifier.streamlit.app/`
#### Live web app may take some time to active. third page (prediction page) `may take 2-3 minutes to for model run/training for the first time run`. thanks for your patients.

1. `Clone the repository:`
```bash
git clone https://github.com/AzadMehedi/Attention-Span-Classifier.git
```

`2. run final.py file from terminal:`
```bash
streamlit run final.py
```
`3. Explore the Streamlit web app.`




---
<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>

## **`Mehedi Hasan`**  
Data Scientist & Statistical Data/Business Analyst` 

📧 [Email](azadmehedi12121@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/mehediazad/)  


