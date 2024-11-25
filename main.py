import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Original data
data = {
    'Date': ['2023-06-20', '2023-11-14', '2023-12-28', '2024-03-12', '2024-03-27', '2024-04-17', '2024-05-14',
             '2024-07-16', '2024-08-13', '2024-09-25', '2024-10-08', '2024-10-18', '2024-10-23', '2024-10-31',
             '2024-11-13', '2024-11-21'],
    'Enrolled': [1, 7, 12, 21, 23, 24, 31, 38, 42, 46, 48, 51, 52, 56, 59, 60],
    # 'Sites': [1, 5, 5, 5, 5, 6, 8, 8, 9, 9, 11, 11, 12, 12, 12, 12],
}

site_data = {
    'Date': ['2023-04-20', '2023-08-21', '2023-10-24', '2023-10-24', '2023-10-24', '2024-04-29', '2024-04-29',
             '2024-06-10', '2024-06-10', '2024-07-11', '2024-10-07', '2024-10-07', '2024-10-29'],
    'Location': [
        "Corvallis, Oregon, United States, 97330",
        "San Antonio, Texas, United States, 78229",
        "Augusta, Georgia, United States, 30909",
        "Detroit, Michigan, United States, 4820",
        "Ann Arbor, Michigan, United States, 48109",
        "Shreveport, Louisiana, United States, 7110",
        "Aurora, Colorado, United States, 80045",
        "Cincinnati, Ohio, United States, 45219",
        "San Antonio, Texas, United States, 78229",
        "Iowa City, Iowa, United States, 5224",
        "Fort Sam Houston, Texas, United States, 78234",
        "JBSA Fort Sam Houston, Texas, United States, 78234",
        "Palo Alto, California, United States, 94304"
    ],
    'Sites': [1, 2, 5, 7, 9, 9, 11, 12, 12, 12,12,12,12]
}

# Convert data to DataFrame
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

site_df = pd.DataFrame(site_data)
site_df['Date'] = pd.to_datetime(site_df['Date'])

# Polynomial regression to model enrollment
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[['Days']])
model = LinearRegression()
model.fit(X_poly, df['Enrolled'])

# Future projections
future_days = np.arange(df['Days'].max(), df['Days'].max() + 365, 30)
future_dates = pd.date_range(start=df['Date'].max(), periods=len(future_days), freq='30D')
future_days_poly = poly.transform(future_days.reshape(-1, 1))
future_enrollment = model.predict(future_days_poly)

# Cap enrollment at 200
future_enrollment = np.clip(future_enrollment, a_min=0, a_max=200)
cutoff_index = np.argmax(future_enrollment >= 200)
if cutoff_index > 0:
    future_days = future_days[:cutoff_index + 1]
    future_dates = future_dates[:cutoff_index + 1]
    future_enrollment = future_enrollment[:cutoff_index + 1]

# Site projections logic: 1 site added every 2 months until a cap of 20 sites
current_sites = site_df['Sites'].iloc[-1]
max_sites = 20
new_sites = np.minimum(current_sites + np.arange(len(future_days)) // 2, max_sites)

# Adjust enrollment based on site growth
def enrollment_with_sites(base_enrollment, sites):
    return base_enrollment * (sites / current_sites)

adjusted_enrollment = enrollment_with_sites(future_enrollment, new_sites)

# Identify points where new sites are added
site_addition_points = np.where(np.diff(new_sites) > 0)[0] + 1
site_addition_dates = future_dates[site_addition_points]
site_addition_values = adjusted_enrollment[site_addition_points]

# Group site data by Date and aggregate locations into a single string
grouped_site_data = site_df.groupby('Date')['Location'].apply(lambda x: '<br>'.join(x)).reset_index()

# Interactive plot
fig = go.Figure()

# Historical enrollment
fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Enrolled'],
    mode='lines+markers',
    name='Actual Enrollment',
    hovertemplate="Date: %{x}<br>Enrollment: %{y}<extra></extra>",
    line=dict(color='blue')
))

# Future enrollment projections
fig.add_trace(go.Scatter(
    x=future_dates,
    y=adjusted_enrollment,
    mode='lines',
    name='Adjusted Projected Enrollment',
    hovertemplate="Date: %{x}<br>Projected Enrollment: %{y:.0f}<extra></extra>",
    line=dict(color='orange', dash='dot')
))

# Actual site locations with hover information
# fig.add_trace(go.Scatter(
#     x=grouped_site_data['Date'],  # Use the grouped dates
#     y=[0] * len(grouped_site_data),  # Set y=0 for hover display
#     mode='markers',
#     name='Site Locations',
#     hovertemplate="Date: %{x}<br>Locations:<br>%{text}<extra></extra>",
#     marker=dict(size=8, color='green', symbol='circle'),
#     text=grouped_site_data['Location']  # Aggregate locations for hover display
# ))

# Historical sites data (number of sites)
# fig.add_trace(go.Scatter(
#     x=site_df['Date'],
#     y=site_df['Sites'],
#     mode='lines+markers',
#     name='Actual Sites',
#     hovertemplate="Date: %{x}<br>Sites: %{y}<extra></extra>",
#     line=dict(color='green', dash='solid')
# ))

# Historical sites data (number of sites) and Actual site locations with hover information
fig.add_trace(go.Scatter(
    x=grouped_site_data['Date'],  # Use the grouped dates
    y=site_df['Sites'],
    mode='lines+markers',
    name='Actual Sites',
    hovertemplate="Date: %{x}<br>Sites: %{y}<br>Locations:<br>%{text}<extra></extra>",
    line=dict(color='green', dash='solid'),
    text = grouped_site_data['Location']  # Aggregate locations for hover display
))

# Plot site additions as red circles
fig.add_trace(go.Scatter(
    x=site_addition_dates,
    y=site_addition_values,
    mode='markers',
    name='New Site Added',
    marker=dict(color='red', size=10, symbol='x'),
    hovertemplate="Site Added: %{y:.0f}<br>Date: %{x}<extra></extra>"
))

# Formatting
fig.update_layout(
    title="Enrollment Projections with Site Growth (Capped at 200 Patients)",
    xaxis_title="Date",
    yaxis_title="Enrollment / Sites",
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

fig.show()
