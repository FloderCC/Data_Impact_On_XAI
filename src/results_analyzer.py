import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.dataset_utils import dataset_description_header


def plot_mcc_and_create_dataframe_with_the_correlations():
    # opening the csv file
    df = pd.read_csv('results/results.csv')

    # csv file columns
    # Seed,Dataset,Sample Size,Number of training samples,Number of testing samples,Number of features,Class Imbalance Ratio,Gini Impurity,Entropy,Completeness,Consistency,Uniqueness,Redundancy (avg),Redundancy (std),Avg of features' avg,Std of features' avg,Avg of features' std,Std of features' std,Model,MCC,XAI,Train F Relevance,Test F Relevance

    unique_seed_values = df['Seed'].unique()
    unique_dataset_values = df['Dataset'].unique()
    unique_sample_size_values = df['Sample Size'].unique()
    unique_model_values = df['Model'].unique()

    for seed in unique_seed_values:
        for dataset in unique_dataset_values:
            for sample_size in unique_sample_size_values:
                for model in unique_model_values:
                    df_filtered = df[
                        (df['Seed'] == seed) & (df['Dataset'] == dataset) & (df['Sample Size'] == sample_size) & (
                                df['Model'] == model)]

                    if len(df_filtered) != 0:
                        pi_values = df_filtered[(df_filtered['XAI'] == 'PI')]['Test F Relevance'].values[0]
                        shap_values = df_filtered[(df_filtered['XAI'] == 'SHAP')]['Test F Relevance'].values[0]
                        lime_values = df_filtered[(df_filtered['XAI'] == 'LIME')]['Test F Relevance'].values[0]

                        pi_values = [float(x) for x in pi_values[1:-1].split(', ')]
                        shap_values = [float(x) for x in shap_values[1:-1].split(', ')]
                        lime_values = [float(x) for x in lime_values[1:-1].split(', ')]


                        # updating the column  df in the row corresponding to the first index of df_filtered
                        df.at[df_filtered.index[0], 'XAI pair'] = 'PI - SHAP'
                        df.at[df_filtered.index[0], 'XAI PCC'] = pd.Series(pi_values).corr(pd.Series(shap_values))
                        df.at[df_filtered.index[0], 'XAI SRCC'] = pd.Series(pi_values).corr(pd.Series(shap_values), method='spearman')

                        df.at[df_filtered.index[1], 'XAI pair'] = 'PI - LIME'
                        df.at[df_filtered.index[1], 'XAI PCC'] = pd.Series(pi_values).corr(pd.Series(lime_values))
                        df.at[df_filtered.index[1], 'XAI SRCC'] = pd.Series(pi_values).corr(pd.Series(lime_values), method='spearman')

                        df.at[df_filtered.index[2], 'XAI pair'] = 'SHAP - LIME'
                        df.at[df_filtered.index[2], 'XAI PCC'] = pd.Series(shap_values).corr(pd.Series(lime_values))
                        df.at[df_filtered.index[2], 'XAI SRCC'] = pd.Series(shap_values).corr(pd.Series(lime_values), method='spearman')

    # dropping columns that are not necessary: XAI, Test F Relevance
    df.drop(columns=['XAI', 'Test F Relevance'], inplace=True)
    df.to_csv('results/correlation_results.csv', index=False)


    sns.set_theme(style="whitegrid")
    # note: sample size and seed are the errors

    plt.figure(figsize=(5.8, 3.7))
    # setting font size
    # plt.rcParams.update({'font.size': 16})
    ax = sns.barplot(x='Dataset', y='MCC', data=df, errorbar='sd', hue='Model',
                palette='tab20', capsize=0.2, width=0.85)

    # add horizontal line 0.8 MCC as a cutoff (dashed)
    plt.axhline(y=0.8, color='#545353', linewidth=1.0, linestyle='--')

    fontsize = 12

    # Create a custom legend and center it
    ax.legend(fontsize=fontsize, loc='upper right', bbox_to_anchor=(1.45, 1))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set axis labels font size
    ax.set_xlabel('Dataset', fontsize=fontsize+3)
    ax.set_ylabel('MCC', fontsize=fontsize+3)
    # Set tick labels font size
    ax.tick_params(axis='both', which='major', labelsize=fontsize+3)



    plt.savefig(f'plots/png/1_error_MCC.png', bbox_inches='tight')
    plt.savefig(f'plots/pdf/1_error_MCC.pdf', bbox_inches='tight')

plot_mcc_and_create_dataframe_with_the_correlations()

def load_dataframe_with_the_correlations_and_plot():  # SRCC
    df = pd.read_csv('results/correlation_results.csv')

    data_to_plot = {
        'hp': {
            # models with mcc >= 0.8
            'df': df[(df['MCC'] >= 0.8)],
        },
        'lp': {
            # models with mcc < 0.8
            'df': df[(df['MCC'] < 0.8)],
        }
    }

    for performance, data in data_to_plot.items():
        df_to_plot = data['df']
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(6, 3.7))
        ax = sns.barplot(x='XAI pair', y=f'XAI SRCC', data=df_to_plot, errorbar='sd', hue='Dataset', legend=True, palette='tab20', capsize=0.2)
        # ax.set_title(correlation_type)
        ax.set_xlabel('XAI Pair')
        ax.set_ylabel(f"SRCC Correlation")

        fontsize = 11

        # Set axis labels font size
        ax.set_xlabel('XAI Pair', fontsize=fontsize)
        ax.set_ylabel('SRCC Correlation', fontsize=fontsize)
        # Set tick labels font size
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        ax.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)

        # set y top limit to 1.2
        ax.set_ylim([-0.03, 1.0])

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.savefig(f'plots/png/2_correlations_{performance}_xai.png', bbox_inches='tight')
        plt.savefig(f'plots/pdf/2_correlations_{performance}_xai.pdf', bbox_inches='tight')

# load_dataframe_with_the_correlations_and_plot()

def load_dataframe_with_the_correlations_and_plot_data_impact():
    df_o = pd.read_csv('results/correlation_results.csv')

    data_to_plot = {
        'hp': df_o[(df_o['MCC'] >= 0.8)],
        'lp': df_o[(df_o['MCC'] < 0.8)]
    }

    for performance, df in data_to_plot.items():
        input_features = dataset_description_header
        # plot the correlation between the input_features, and each output_feature. Each plot has two subplots: one for Pearson correlation (at the left) and one for Spearman correlation (at the right). The error is given by the "XAI Pair" column. The y-axis is the input feature and the x-axis is the column "Correlation Pearson".
        corr_dataframe = pd.DataFrame(columns=['Input Feature', 'SRCC', 'XAI Pair'])
        for input_feature in input_features:
            for xai_pair in df['XAI pair'].unique():
                correlation_pearson_value = df[(df['XAI pair'] == xai_pair)][input_feature].corr(df[(df['XAI pair'] == xai_pair)]["XAI SRCC"])
                correlation_spearman_value = df[(df['XAI pair'] == xai_pair)][input_feature].corr(df[(df['XAI pair'] == xai_pair)]["XAI SRCC"], method='spearman')
                corr_dataframe = corr_dataframe._append({'Input Feature': input_feature, 'SRCC': correlation_spearman_value, 'XAI Pair': xai_pair}, ignore_index=True)

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(4, 6))

        correlation_type = 'SRCC'
        legend = True

        ax = sns.barplot(x=correlation_type, y='Input Feature', data=corr_dataframe, errorbar='sd', hue='XAI Pair',
                         legend=legend, palette='tab20', dodge=True)


        # Add vertical lines to divide groups
        # Identify unique categories and calculate y positions for horizontal lines
        unique_features = corr_dataframe['Input Feature'].unique()
        group_boundaries = [i for i in range(len(unique_features) - 1)]
        # Add horizontal lines to divide groups
        for boundary in group_boundaries:
            y_value = boundary + 0.5  # Adjust to fit your plot's y-axis scale
            ax.axhline(y=y_value, color='gray', linestyle='dashed', linewidth=0.5)

        ax.set_xlim([-0.7, 0.7])

        fontsize = 9

        # Set axis labels font size
        ax.set_xlabel('SRCC Correlation', fontsize=fontsize)
        ax.set_ylabel('Dataset characteristic', fontsize=fontsize)
        # Set tick labels font size
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        # Add legend
        if legend:
            # ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1)).set_zorder(100)
            ax.legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)


        plt.savefig(f'plots/png/3_correlations_{performance}_meta.png', bbox_inches='tight')
        plt.savefig(f'plots/pdf/3_correlations_{performance}_meta.pdf', bbox_inches='tight')

# load_dataframe_with_the_correlations_and_plot_data_impact()




