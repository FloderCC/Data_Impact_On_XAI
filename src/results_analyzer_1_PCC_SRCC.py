import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
                        pi_values = df_filtered[(df_filtered['XAI'] == 'Permutation Importance')]['Test F Relevance'].values[0]
                        shap_values = df_filtered[(df_filtered['XAI'] == 'SHAP')]['Test F Relevance'].values[0]
                        lime_values = df_filtered[(df_filtered['XAI'] == 'LIME (ALL)')]['Test F Relevance'].values[0]

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

    plt.figure(figsize=(10, 6))
    # setting font size
    plt.rcParams.update({'font.size': 18})
    ax = sns.barplot(x='Dataset', y='MCC', data=df, errorbar='sd', hue='Model',
                palette='tab20')


    # Create a custom legend and center it
    ax.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.30, 1))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # add horizontal line 0.8 MCC as a cutoff (dashed)
    plt.axhline(y=0.8, color='#545353', linewidth=1.0, linestyle='--')
    plt.savefig(f'plots/1_error_MCC.png', bbox_inches='tight')


# plot_mcc_and_create_dataframe_with_the_correlations()
def load_dataframe_with_the_correlations_and_plot():
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
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        # setting font size
        plt.rcParams.update({'font.size': 18})
        for i, correlation_type in enumerate(['PCC', 'SRCC']):
            ax = sns.barplot(x='XAI pair', y=f'XAI {correlation_type}', data=df_to_plot, errorbar='sd', hue='Dataset', legend=True, ax=axs[i], palette='tab20')
            # ax.set_title(correlation_type)
            ax.set_xlabel('XAI Pair')
            ax.set_ylabel(f"{correlation_type} Correlation")
            # set y top limit to 1.2
            ax.set_ylim([0, 1.2])
        plt.savefig(f'plots/2_correlations_{performance}.png', bbox_inches='tight')

load_dataframe_with_the_correlations_and_plot()

def load_dataframe_with_the_correlations_and_plot_data_impact():
    df = pd.read_csv('results/correlation_results.csv')

    input_features = ["Number of training samples", "Number of testing samples", "Number of features", "Class Imbalance Ratio", "Gini Impurity", "Entropy", "Completeness", "Consistency", "Uniqueness", "Redundancy (avg)", "Redundancy (std)", "Avg of features' avg", "Std of features' avg", "Avg of features' std", "Std of features' std"]
    output_features = ["XAI PCC", "XAI SRCC"]
    # plot the correlation between the input_features, and each output_feature. Each plot has two subplots: one for Pearson correlation (at the left) and one for Spearman correlation (at the right). The error is given by the "XAI Pair" column. The y-axis is the input feature and the x-axis is the column "Correlation Pearson".
    for output_feature in output_features:
        # creating a new dataframe with the correlation values between the input features and the output feature
        corr_dataframe = pd.DataFrame(columns=['Input Feature', 'PCC', 'SRCC', 'XAI Pair'])
        for input_feature in input_features:
            for xai_pair in df['XAI pair'].unique():
                correlation_pearson_value = df[(df['XAI pair'] == xai_pair)][input_feature].corr(df[(df['XAI pair'] == xai_pair)][output_feature])
                correlation_spearman_value = df[(df['XAI pair'] == xai_pair)][input_feature].corr(df[(df['XAI pair'] == xai_pair)][output_feature], method='spearman')
                corr_dataframe = corr_dataframe._append({'Input Feature': input_feature, 'PCC': correlation_pearson_value, 'SRCC': correlation_spearman_value, 'XAI Pair': xai_pair}, ignore_index=True)



        # plot the correlation values between the dataset characteristics and XAI Pair correlation values
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        # setting font size
        # plt.rcParams.update({'font.size': 18})
        for i, correlation_type in enumerate(['PCC', 'SRCC']):
            legend = False
            if i == 1:
                legend = True

            ax = sns.barplot(x=correlation_type, y='Input Feature', data=corr_dataframe, errorbar='sd', hue='XAI Pair', legend=legend, ax=axs[i], palette='tab20')
            ax.set_xlabel(correlation_type)
            ax.set_ylabel('Dataset Characteristic')

            # set ax font size
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(16)

            if legend:
                # put the legend on front of all the other elements

                ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1)).set_zorder(100)



        # removing the y labels, legend, and y ticks from the second plot
        axs[1].set_ylabel('')
        axs[1].set_yticks([])



        plt.savefig(f'plots/3_correlations_{output_feature}.png', bbox_inches='tight')

# load_dataframe_with_the_correlations_and_plot_data_impact()

















    # # correlations plots between dataset features and outputs
    # inputs = ["Number of training samples", "Number of testing samples", "Number of features", "Class Imbalance Ratio", "Gini Impurity", "Entropy", "Completeness", "Consistency", "Uniqueness", "Redundancy (avg)", "Redundancy (std)", "Avg of features' avg", "Std of features' avg", "Avg of features' std", "Std of features' std"]
    # outputs = ["Pearson Correlation PI - SHAP", "Pearson Correlation PI - LIME", "Pearson Correlation SHAP - LIME", "Spearman Correlation PI - SHAP", "Spearman Correlation PI - LIME", "Spearman Correlation SHAP - LIME"]
    #
    # correlation_type_list = ['Pearson', 'Spearman']
    # correlation_pair_list = ['PI - SHAP', 'PI - LIME', 'SHAP - LIME']
    # custom_palette_for_inputs = {feature: color for feature, color in
    #                   zip(inputs, sns.color_palette("tab20", n_colors=len(inputs)))}
    # for correlation_pair in correlation_pair_list:
    #     # Plot horizontal bars for each correlation type. Each plot has two subplots: one for Pearson (at the left) and one for Spearman (at the right).
    #     fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    #     fig.suptitle(f'Correlation values between {correlation_pair} for each input feature')
    #     for i, correlation_type in enumerate(correlation_type_list):
    #         ax = sns.barplot(x=correlation_type + ' Correlation ' + correlation_pair, y='Model', data=df, errorbar='sd', hue='Model', legend=False, ax=axs[i], palette=custom_palette_for_models)
    #         ax.set_title(correlation_type + ' Correlation')
    #         ax.set_xlabel(correlation_type + ' Correlation')
    #         ax.set_ylabel('Model')




