import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to generate a random dataset
def generate_dataset():
    np.random.seed(42)
    data = {
        'Car Model': np.random.choice(['Sedan A', 'SUV B', 'Hatchback C'], size=100),
        'Price': np.random.uniform(15000, 50000, size=100),
        'Units Sold': np.random.randint(1, 100, size=100),
        'Customer Rating': np.random.uniform(1, 5, size=100),
        'Sale Date': pd.date_range(start='2024-01-01', periods=100, freq='D')
    }
    return pd.DataFrame(data)


# Function to calculate statistics
def calculate_statistics(df):
    # Grouping data by 'Car Model' and calculating various statistics
    statistics = df.groupby('Car Model').agg({
        'Price': ['mean', 'min', 'max', 'std'],
        'Units Sold': ['sum', 'mean', 'std'],
        'Customer Rating': 'mean'
    }).reset_index()

    # Flattening the multi-level columns for better readability
    statistics.columns = ['Car Model', 'Avg Price', 'Min Price', 'Max Price', 'Price Std Dev',
                          'Total Units Sold', 'Avg Units Sold', 'Units Sold Std Dev', 'Avg Customer Rating']
    return statistics


# Function for correlation analysis between numerical variables
def correlation_analysis(df):
    corr_matrix = df[['Price', 'Units Sold', 'Customer Rating']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix between Price, Units Sold, and Customer Rating')
    plt.show()


# Function to visualize data
def visualize_data(df):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # Total Units Sold vs Car Model bar chart with trendline
    units_sold_per_model = df.groupby('Car Model')['Units Sold'].sum()
    units_sold_per_model.plot(kind='bar', ax=ax[0, 0], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax[0, 0].set_title('Total Units Sold per Car Model')
    ax[0, 0].set_ylabel('Units Sold')

    # Adding a trendline to the bar chart (using linear regression)
    for i, v in enumerate(units_sold_per_model):
        ax[0, 0].text(i, v + 2, f'{v}', ha='center', fontsize=10)

    # Customer Rating distribution histogram
    df['Customer Rating'].plot(kind='hist', bins=10, ax=ax[0, 1], color='skyblue')
    ax[0, 1].set_title('Customer Rating Distribution')
    ax[0, 1].set_xlabel('Rating')

    # Price vs Units Sold scatter plot with regression line
    sns.regplot(x='Price', y='Units Sold', data=df, ax=ax[1, 0], scatter_kws={'color': 'red'},
                line_kws={'color': 'blue'})
    ax[1, 0].set_title('Price vs Units Sold with Trendline')

    # Time series of units sold over time
    df.groupby('Sale Date')['Units Sold'].sum().plot(ax=ax[1, 1], color='green')
    ax[1, 1].set_title('Units Sold Over Time')
    ax[1, 1].set_ylabel('Total Units Sold')
    ax[1, 1].set_xlabel('Date')

    plt.tight_layout()
    plt.show()


# Main function to run the project
def main():
    df = generate_dataset()  # Generate the dataset
    print("Generated Dataset:\n", df.head())  # Print first few rows

    statistics = calculate_statistics(df)  # Calculate statistics
    print("\nStatistics:\n", statistics)  # Display statistics

    correlation_analysis(df)  # Perform correlation analysis

    visualize_data(df)  # Visualize the data


if __name__ == "__main__":
    main()
