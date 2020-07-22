#include "include/mcm/pass/graphOptimizations/StrategyRegistry.hpp"

namespace mv {
namespace graphOptimizer {

//################## DEFAULT GLOBAL CONFIG'S FOR Kmb ####################
MV_OPTIMIZER_GLOBAL_CONFIG_REGISTRY()
    .enter("referenceDevice").set("A0");

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

//##################DEFAULT GLOBAL STRATEGIES FOR Kmb ###################

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("forceSpilling").set(true);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("enableStreaming").set(true);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("doubleBuffering").set(false);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("enableSparsity").set(false);

MV_OPTIMIZER_GLOBAL_STRATEGY_REGISTRY()
    .enter("clusteringStrategy").set("Automatic");

//################# DEFAULT LAYER STRATEGIES FOR Kmb #####################

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("Conv")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH","StreamOverK","StreamOverN"})
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverH", "SplitOverK"})
    .registerSet("inputActivationSparsity").insert(true)
    .registerSet("outputActivationSparsity").insert(true)
    .registerSet("weightsSparsity").insert(true);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("DepthwiseConv")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH", "StreamOverC"})
    .registerSet("ClusteringStrategies").insert(vector<string>{"Clustering","SplitOverH"})
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(true)
    .registerSet("weightsSparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("MaxPool")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH"})
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverH","HKSwitch"})
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(true)
    .registerSet("weightsSparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("Eltwise")
    .registerSet("streamingStrategies").insert(vector<string>{"StreamOverH"})
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering","SplitOverH","HKSwitch"})
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(true)
    .registerSet("weightsSparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("Concat")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering"})
    .registerSet("forceSpilling").insert(true)
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(false)
    .registerSet("weightsSparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("Input")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering", "SplitOverK", "SplitOverH", "SplitOverHOverlapped"})
    .registerSet("forceSpilling").insert(false)
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(false)
    .registerSet("weightsSparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("ImplicitInput")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering", "SplitOverK", "SplitOverH", "SplitOverHOverlapped"})
    .registerSet("forceSpilling").insert(false)
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(false)
    .registerSet("weightsSparsity").insert(false);

MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
    .enter("Output")
    .registerSet("streamingStrategies").insert(vector<string>(0))
    .registerSet("clusteringStrategies").insert(vector<string>{"Clustering", "SplitOverH", "SplitOverK"})
    .registerSet("forceSpilling").insert(true)
    .registerSet("inputActivationSparsity").insert(false)
    .registerSet("outputActivationSparsity").insert(false)
    .registerSet("weightsSparsity").insert(false);



//################# DEFAULT SW layers #####################
MV_OPTIMIZER_LAYER_STRATEGY_REGISTRY()
   .enter("Default")
   .registerSet("streamingStrategies").insert(vector<string>(0))
   .registerSet("clusteringStrategies").insert(vector<string>{"Clustering"})
   .registerSet("forceSpilling").insert(true)
   .registerSet("inputActivationSparsity").insert(false)
   .registerSet("outputActivationSparsity").insert(false)
   .registerSet("weightsSparsity").insert(false);

}
}
