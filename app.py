import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set page config
st.set_page_config(
    page_title="Gun-Related Suicide Analysis",
    page_icon="🔍",
    layout="wide"
)


# Define helper functions
def load_data():
    """Load the cleaned data for analysis"""
    try:
        data = pd.read_csv('cleaned_gun_deaths.csv')
    except FileNotFoundError:
        # If the cleaned file doesn't exist, load and clean the original data
        data = pd.read_csv('gun_deaths.csv')
        data = clean_data(data)
    
    return data


def clean_data(data):
    """Clean the original dataset"""
    # Drop duplicate rows
    data = data.drop_duplicates()
    
    # Drop rows with missing values in key columns
    for col in ['intent', 'age', 'place', 'education']:
        data = data.dropna(subset=[col])
    
    # Convert categorical columns to category type
    cat_cols = ['year', 'month', 'police', 'sex', 'race', 'place', 'education']
    for col in cat_cols:
        data[col] = data[col].astype('category')
    
    # Save the cleaned data
    data.to_csv('cleaned_gun_deaths.csv', index=False)
    return data


def create_age_groups(data):
    """Create age groups for analysis"""
    age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 100]
    age_labels = ['0-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    
    data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)
    return data


# App title and introduction
st.title("Analysis of Gun-Related Suicides in the United States")
st.markdown("""
This dashboard presents an analysis of gun-related suicides in the United States, focusing on three key hypotheses:
1. **Suicide rates are higher for individuals with lower education levels**
2. **Gun-related suicides peak in winter months**
3. **Gun-related suicide tends to increase as age increases**
""")

# Load data
with st.spinner("Loading data..."):
    data = load_data()
    data_with_age_groups = create_age_groups(data)
    suicide_data = data[data['intent'] == 'Suicide']

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", 
    "Hypothesis 1: Education", 
    "Hypothesis 2: Seasonality", 
    "Hypothesis 3: Age"
])

# Tab 1: Overview
with tab1:
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"Total records: {len(data):,}")
        st.write(f"Suicide cases: {len(suicide_data):,}")
        st.write(f"Time period: {data['year'].min()} to {data['year'].max()}")
    
    with col2:
        st.subheader("Intent Distribution")
        intent_counts = data['intent'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=intent_counts.index, y=intent_counts.values, palette='viridis')
        plt.title('Distribution of Death Intents')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Display sample of the data
    st.subheader("Sample Data")
    st.dataframe(data.head())

# Tab 2: Hypothesis 1 - Education Levels
with tab2:
    st.header("Hypothesis 1: Suicide rates are higher for individuals with lower education levels")
    
    # Filter suicide data
    education_suicide_data = suicide_data['education'].value_counts().sort_index()
    
    # Calculate percentages
    total_suicides = education_suicide_data.sum()
    education_percentages = (education_suicide_data / total_suicides) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create contingency table
        education_intent_table = pd.crosstab(data['education'], data['intent'])
        st.subheader("Suicide Counts by Education Level")
        st.dataframe(pd.DataFrame({
            'Education Level': education_suicide_data.index,
            'Suicide Count': education_suicide_data.values,
            'Percentage': education_percentages.values.round(1)
        }).sort_values('Suicide Count', ascending=False))
        
        # Chi-square test results
        st.subheader("Statistical Test Results")
        chi2, p, dof, expected = stats.chi2_contingency(education_intent_table)
        st.write(f"Chi-square statistic: {chi2:.4f}")
        st.write(f"p-value: {p:.8e}")
        conclusion = "Significant difference in suicide rates across education levels" if p < 0.05 else "No significant difference in suicide rates across education levels"
        st.write(f"Conclusion: {conclusion}")
    
    with col2:
        # Create visualization
        st.subheader("Distribution of Suicides by Education Level")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("viridis", len(education_suicide_data))
        plt.pie(
            education_suicide_data, 
            labels=education_suicide_data.index, 
            autopct='%1.1f%%', 
            startangle=90, 
            shadow=True, 
            colors=colors
        )
        plt.axis('equal')
        plt.title('Distribution of Suicides by Education Level', fontsize=16)
        st.pyplot(fig)
    
    # Interactive visualization section
    st.subheader("Interactive Analysis: Education Level vs. Intent")
    
    # Calculate the percentage of each intent within each education level
    intent_by_education = pd.crosstab(
        data['education'], 
        data['intent'], 
        normalize='index'
    ) * 100
    
    # Add option to select which intents to display
    st.write("Select intents to display:")
    available_intents = list(intent_by_education.columns)
    selected_intents = st.multiselect(
        "Intent types", 
        available_intents,
        default=["Suicide"]
    )
    
    if not selected_intents:
        st.warning("Please select at least one intent type.")
    else:
        # Filter data based on selection
        filtered_intent_data = intent_by_education[selected_intents]
        
        # Create the interactive bar chart
        fig = plt.figure(figsize=(10, 6))
        
        # Plot each selected intent
        for intent in selected_intents:
            plt.bar(
                filtered_intent_data.index,
                filtered_intent_data[intent],
                label=intent
            )
        
        plt.title(f"Percentage of Selected Intents by Education Level", fontsize=16)
        plt.xlabel("Education Level", fontsize=12)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Add an interactive slider for comparing education levels
        st.subheader("Compare Education Levels")
        education_options = list(education_suicide_data.index)
        education_to_compare = st.select_slider(
            "Select education levels to compare:",
            options=education_options,
            value=(education_options[0], education_options[-1])
        )
        
        # Create comparison visualization
        if education_to_compare[0] != education_to_compare[1]:
            # Extract the two selected education levels
            edu1, edu2 = education_to_compare
            
            # Get data for the selected education levels
            comparison_data = intent_by_education.loc[[edu1, edu2]]
            
            # Create the comparison chart
            fig, ax = plt.subplots(figsize=(10, 5))
            comparison_data.plot(kind='bar', ax=ax)
            plt.title(f"Intent Distribution: {edu1} vs {edu2}", fontsize=16)
            plt.xlabel("Education Level", fontsize=12)
            plt.ylabel("Percentage (%)", fontsize=12)
            plt.legend(title="Intent")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # Add text explanation
            if "Suicide" in intent_by_education.columns:
                suicide_diff = intent_by_education.loc[edu1, "Suicide"] - intent_by_education.loc[edu2, "Suicide"]
                direction = "higher" if suicide_diff > 0 else "lower"
                st.write(f"**Insight:** The {edu1} education group has a {abs(suicide_diff):.1f}% {direction} suicide percentage compared to the {edu2} education group.")

# Tab 3: Hypothesis 2 - Seasonality
with tab3:
    st.header("Hypothesis 2: Gun-related suicides peak in winter months")
    
    # Define winter months and seasons
    winter_months = [12, 1, 2]
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    
    # Count suicides by month
    monthly_suicides = suicide_data.groupby('month').size()
    
    # Create a DataFrame for visualization
    monthly_suicide_df = pd.DataFrame({
        'Month': range(1, 13),
        'Suicide_Count': [monthly_suicides.get(i, 0) for i in range(1, 13)]
    })
    
    # Add a season column and calculate daily averages
    def get_season(month):
        if month in winter_months:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    monthly_suicide_df['Season'] = monthly_suicide_df['Month'].apply(get_season)
    monthly_suicide_df['Days_in_Month'] = monthly_suicide_df['Month'].map(days_in_month)
    monthly_suicide_df['Daily_Average'] = monthly_suicide_df['Suicide_Count'] / monthly_suicide_df['Days_in_Month']
    
    # Create seasonal aggregation
    season_data = monthly_suicide_df.groupby('Season')[['Suicide_Count', 'Days_in_Month']].sum()
    season_data['Daily_Average'] = season_data['Suicide_Count'] / season_data['Days_in_Month']
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_data = season_data.reindex(season_order)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Suicide Data")
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        display_df = monthly_suicide_df.assign(Month_Name=monthly_suicide_df['Month'].map(month_names))
        st.dataframe(display_df[['Month_Name', 'Suicide_Count', 'Daily_Average', 'Season']])
        
        st.subheader("Seasonal Suicide Data")
        st.dataframe(season_data)
        
        # Statistical test
        st.subheader("Statistical Test Results")
        winter_daily_avg = monthly_suicide_df[monthly_suicide_df['Month'].isin(winter_months)]['Daily_Average']
        non_winter_months = [m for m in range(1, 13) if m not in winter_months]
        non_winter_daily_avg = monthly_suicide_df[monthly_suicide_df['Month'].isin(non_winter_months)]['Daily_Average']
        
        t_stat, p_value = stats.ttest_ind(winter_daily_avg, non_winter_daily_avg, equal_var=False)
        st.write(f"T-test statistic: {t_stat:.4f}")
        st.write(f"p-value: {p_value:.8f}")
        st.write(f"Average daily suicides in winter months: {winter_daily_avg.mean():.2f}")
        st.write(f"Average daily suicides in non-winter months: {non_winter_daily_avg.mean():.2f}")
        
        if p_value < 0.05:
            conclusion = "Significant difference in suicide rates between winter and non-winter months"
            if winter_daily_avg.mean() > non_winter_daily_avg.mean():
                conclusion += ". Winter months have higher rates."
            else:
                conclusion += ". Winter months have lower rates."
        else:
            conclusion = "No significant difference in suicide rates between winter and non-winter months"
        st.write(f"Conclusion: {conclusion}")
    
    with col2:
        # Create monthly trend visualization
        st.subheader("Average Daily Gun-Related Suicides by Month")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Month', y='Daily_Average', data=monthly_suicide_df, marker='o', linewidth=2)
        plt.title('Average Daily Gun-Related Suicides by Month', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Average Daily Suicide Count', fontsize=12)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Create seasonal bar chart
        st.subheader("Average Daily Gun-Related Suicides by Season")
        fig, ax = plt.subplots(figsize=(10, 6))
        season_palette = ['#81D4FA', '#A5D6A7', '#F48FB1', '#FFCC80']
        sns.barplot(x=season_data.index, y=season_data['Daily_Average'], palette=season_palette)
        plt.title('Average Daily Gun-Related Suicides by Season', fontsize=16)
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Average Daily Suicide Count', fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

# Tab 4: Hypothesis 3 - Age
with tab4:
    st.header("Hypothesis 3: Gun-related suicide tends to increase as age increases")
    
    # Create age group contingency table
    age_group_table = pd.crosstab(data_with_age_groups['age_group'], data_with_age_groups['intent'])
    
    # Calculate suicide counts and rates by age group
    suicide_by_age = age_group_table['Suicide']
    total_by_age = age_group_table.sum(axis=1)
    suicide_rate_by_age = (suicide_by_age / total_by_age) * 100
    
    # Create DataFrame for visualization
    age_labels = ['0-17', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    age_suicide_df = pd.DataFrame({
        'Age Group': age_labels,
        'Suicide Count': suicide_by_age.values,
        'Suicide Rate (%)': suicide_rate_by_age.values,
        'Total Deaths': total_by_age.values,
        'Population Distribution (%)': (total_by_age / total_by_age.sum()) * 100
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Suicide Statistics by Age Group")
        st.dataframe(age_suicide_df)
        
        # Chi-square test results
        st.subheader("Statistical Test Results")
        chi2_age, p_age, dof_age, expected_age = stats.chi2_contingency(age_group_table)
        st.write(f"Chi-square statistic: {chi2_age:.4f}")
        st.write(f"p-value: {p_age:.8e}")
        conclusion = "Significant difference in suicide rates across age groups" if p_age < 0.05 else "No significant difference in suicide rates across age groups"
        st.write(f"Conclusion: {conclusion}")
    
    with col2:
        # Create visualization of suicide counts
        st.subheader("Number of Gun-Related Suicides by Age Group")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Age Group', y='Suicide Count', data=age_suicide_df, palette='viridis')
        plt.title('Number of Gun-Related Suicides by Age Group', fontsize=16)
        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Number of Suicides', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Create visualization of suicide rates
        st.subheader("Suicide Rate (%) Within Each Age Group")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Age Group', y='Suicide Rate (%)', data=age_suicide_df, 
                    marker='o', markersize=10, linewidth=3, color='#FF5733')
        plt.title('Suicide Rate (%) Within Each Age Group', fontsize=16)
        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Suicide Rate (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

# Add footer with conclusions
st.markdown("---")
st.subheader("Key Findings and Conclusions")
st.markdown("""
1. **Education Level**: The analysis shows a significant difference in suicide rates across education levels, 
   with individuals having lower education levels showing higher suicide rates.

2. **Seasonality**: There is no significant difference in suicide rates between winter and non-winter months, 
   suggesting that gun-related suicides do not peak specifically in winter.

3. **Age Effect**: There is a significant difference in suicide rates across age groups, with the data 
   supporting the hypothesis that gun-related suicides tend to increase with age.
""")

# Add policy recommendations
st.subheader("Policy Recommendations")
st.markdown("""
- **Education-Focused Interventions**: Implement targeted educational initiatives and mental health 
  support in communities with lower educational attainment.
  
- **Year-Round Prevention**: Maintain consistent suicide prevention efforts throughout the year, 
  rather than focusing on specific seasons.
  
- **Age-Specific Approaches**: Develop specialized mental health resources for older adults, 
  including counseling services and social support programs to combat loneliness and isolation.
  
- **Further Research**: Conduct additional studies to investigate the root causes of the observed 
  trends in gun-related suicides across different demographic groups.
""")

# Add explanatory notes about the data
st.sidebar.header("About the Data")
st.sidebar.info("""
This dashboard analyzes gun-related deaths in the United States, focusing specifically on suicides. 
The dataset includes variables such as year, month, intent, police involvement, sex, age, race, 
place, and education level.

**Data cleaning steps applied:**
- Removed duplicate records
- Handled missing values
- Categorized age groups for analysis
- Filtered to focus on suicide cases
""")

# Add ethics statement
st.sidebar.header("Ethical Considerations")
st.sidebar.warning("""
This analysis deals with sensitive data related to gun deaths and suicide. All data has been anonymized, 
and the analysis aims to inform public health interventions to reduce suicide rates.

The visualizations and statistics presented here are intended to provide insights for policymakers 
and healthcare professionals, not to stigmatize any particular group.
""")