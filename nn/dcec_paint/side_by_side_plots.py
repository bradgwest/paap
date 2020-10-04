import os

import matplotlib.pyplot as plt
import pandas as pd

CLUSTERS = 3

final_df_path = "/home/dubs/dev/paap/img/{}/df.ndjson".format(CLUSTERS)
final_df = pd.read_json(final_df_path, orient="records", lines=True)

rgb_subplots = final_df[['blue', 'green', 'red']].hist(by=final_df['cluster'] + 1, color=["blue", "green", "red"], density=True, sharex=True, sharey=True)
plt.suptitle("Distribution of RGB Pixel Intensity by Cluster (k={})".format(CLUSTERS), fontsize=16, y=1)
plt.show()
plt.savefig(os.path.join("/tmp", "hist.png"), dpi=300)
plt.close()
