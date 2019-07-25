#include "include/mcm/pass/graphOptimizations/StrategyRegistry.hpp"

namespace mv {
namespace graphOptimizer {

//################## DEFAULT GLOBAL CONFIG'S FOR KEEMBAY ####################
MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY()
    .enter("totalClusters").set(1);

MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY()
    .enter("clusterMemory").set(3584);

MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY()
    .enter("dpuPerCluster").set(1);

MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY()
    .enter("ddrBandwidth").set(1);

MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY()
    .enter("systemClockMhz").set(500);

//##################DEFAULT GLOBAL STRATEGIES FOR KEEMBAY ###################

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("tensorSpilling").set(true);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("enableStreaming").set(true);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("doubleBuffering").set(false);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("enableSparsity").set(false);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("clusteringStrategy").set("Automatic");

//################# DEFAULT LAYER STRATEGIES FOR KEEMBAY #####################

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("Conv")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH","StreamOverW","StreamOverK"})
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverH","SplitOverHOverlapped"})
    .registerSet("sparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("DepthWiseConv")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH","StreamOverW"})
    .registerSet("ClusteringStrategies").insert(vector<string>{"Clustering","SplitOverH","SplitOverHOverlapped"})
    .registerSet("sparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("MaxPool")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH","StreamOverW"})
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverH","HKSwitch"})
    .registerSet("sparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("Eltwise")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH","StreamOverW"})
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverH","HKSwitch"})
    .registerSet("sparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("Input")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverHOverlapped"})
    .registerSet("tensorSpilling").insert(true)
    .registerSet("sparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("Output")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering"})
    .registerSet("tensorSpilling").insert(true)
    .registerSet("sparsity").insert(false);

}
}
