# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

# Gun-Related Suicide Analysis in the United States

## Project Overview

This project aims to analyze gun suicides in the US from a large dataset. The analysis aims at determining trends, patterns, and potential risk factors involved in such incidents. Information obtained in this analysis can be utilized in policy-making, public health intervention, and subsequent research.

Firearm suicide is a huge public health issue in the US, disproportionately adding to the total number of suicides. By employing varying criteria in such areas as demographics, geography of distribution, and timing trends, this research tries to advance understanding on underlying causes and possible precipitators for suicide by gun.

The project avails advanced data analysis techniques and visualization tools to present the findings in an intuitive and informative way. The ultimate goal is to contribute to framing effective strategies and policies for reducing gun-related suicides incidence and improving the overall public's health condition.

Key goals of the project are:

- Identifying at-risk demographic populations who are at greater risk of gun-related suicides.
- Investigating geographic trends to determine areas of higher incidence rates.
- Examining temporal patterns to determine if there are annual or seasonal fluctuations.
- Investigating potential correlations of gun suicides with other socio-economic factors.
- Providing informative insights and recommendations to policymakers, healthcare professionals, and researchers.

By these objectives, the project aims to find insights into the puzzling issue of gun suicides and help efforts to combat this public health issue.

## Dataset Explanation

The dataset employed here is a full dataset of gun death in the United States. It holds various attributes that provide information on each incident. The key columns in the dataset are:

- **Year**: The year of the incident.
- **Month**: The month of the incident.
- **Intent**: The reason of the gun death (e.g., Suicide, Homicide, Accidental, Undetermined).
- **Police**: Whether the police officer was a participant in the incident.
- **Sex**: The sex of the individual (M for male, F for female).
- **Age**: The age of the individual at the time of death.
- **Race**: The race of the individual (e.g., White, Black, Asian/Pacific Islander, Native American/Native Alaskan, Hispanic).
- **Place**: Where the incident occurred (e.g., Home, Street, Other specified).
- **Education**: The level of education of the individual (e.g., Less than HS, HS/GED, Some college, BA+).

This dataset provides a rich source of data for research on trends and patterns in gun suicides, allowing an in-depth examination of demographic, geographic, and temporal dimensions. By applying this data, the project aims to uncover insights that can be applied to inform public health interventions and policy-making.

## Libraries Used

The following libraries were used in this project:

- **Pandas**: For data manipulation and analysis, providing powerful data structures and functions needed to clean and analyze the dataset.
- **NumPy**: For numerical operations, enabling efficient computation and manipulation of numerical data.
- **Matplotlib**: For creating static, animated, and interactive visualizations, helping to illustrate the findings clearly.
- **Seaborn**: For statistical data visualization, offering attractive and inf
ormative statistical graphics.
- **Scikit-learn**: For machine learning and data mining, used to build predictive models and uncover patterns in the data.
- **Statsmodels**: For statistical modeling and hypothesis testing, allowing for in-depth statistical analysis and validation of findings.

## Installation and Setup

To get started with this project, follow the steps below to set up your local environment:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Adam-Ansar/Crime-Data.git
    cd Crime-Data
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the analysis**:
    Follow the instructions in the project documentation to run the analysis scripts and visualize the results.


## Ethics Statement

This project involves handling sensitive data on suicides related to guns. I am committed to handling this data respectfully and confidentially. Analysis is carried out with the view to understanding and addressing a critical public health issue, and not to harm or stigmatize any individuals or groups of individuals. Any conclusions and recommendations are presented in a manner that upholds the well-being and dignity of victims. I adhere to ethical values and norms so that the information is used responsibly and the privacy of individuals is maintained.

## Resources

This project leverages various resources to ensure a comprehensive and accurate analysis:

- **Generative AI**: Utilized for data augmentation and generating insights from large datasets, enhancing the depth and breadth of the analysis, and to also upgrade existing code to further improve efficiency of the data analysis.
- **Learning Management System (LMS) Content**: Leveraged content from Code Institute's LMS to guide the analysis process and ensure best practices in data science and ethical considerations. The LMS provided valuable resources and frameworks that were instrumental in shaping the methodology and approach of this project.

By utilizing these resources, the project aims to provide a thorough and responsible analysis of gun-related suicides in the United States, contributing valuable insights to the field of public health. The combination of advanced analytical tools and ethical considerations ensures that the findings are both robust and respectful of the sensitive nature of the data.

## Acknowledgements

I would also like to extend my particular gratitude to Code Institute for providing the resources and assistance necessary to complete this project. Their comprehensive Learning Management System (LMS) and their expert support have been invaluable in informing our methodology and imposing the highest standards in our research. The knowledge and skills gained under their course have been instrumental in the success of this project.
