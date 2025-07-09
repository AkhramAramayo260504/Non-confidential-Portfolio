import pandas as pd
import numpy as np

# Imaginary dataframes for unit testing
te_rank = 2
cbn_rank = 2
ccm_rank = 2
df_model = 2

# Configurable Parameters
ARATTONE = 15

# Reconfigouration of set
te_ranks = te_rank[["Variable", "Rank"]].rename(columns={"Rank": "TE_Rank"})
cbn_ranks = cbn_rank[["Variable", "Rank"]].rename(columns={"Rank": "CBN_Rank"})
ccm_ranks = ccm_rank[["variable", "rank"]].rename(columns={"variable": "Variable", "rank": "CCM_Rank"})

# Merge of the causality dataframes and ranks
merged_ranks = te_ranks.merge(cbn_ranks, on="Variable").merge(ccm_ranks, on="Variable")
merged_ranks["Sum_Ranks"] = merged_ranks["TE_Rank"] + merged_ranks["CBN_Rank"] + merged_ranks["CCM_Rank"]
total_vars = len(merged_ranks)
merged_ranks["Borda_Score"] = total_vars - merged_ranks["Sum_Ranks"]

# Eschaton ranking
eschaton_rank = merged_ranks[["Variable", "Borda_Score"]].copy()
eschaton_rank = eschaton_rank.sort_values("Borda_Score", ascending=True)
eschaton_rank["Final_Rank"] = eschaton_rank["Borda_Score"].rank(method="min", ascending=True).astype(int)
eschaton_rank.columns = ["Variable", "Borda Score", "Final Rank"]
print(eschaton_rank)

# Eschaton Selection Test
top_variables = eschaton_rank['Variable'].head(ARATTONE).tolist()
df_model = df_model[['TARGET'] + top_variables]