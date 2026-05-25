#!/bin/bash
# run_hypotheses_benchmark.sh
# Automated runner to execute hypotheses back-to-back and push results to GitHub.

set -e

# Ensure we are in the correct directory
cd "$(dirname "$0")"

# Create results directory if it doesn't exist
RESULTS_DIR="dissertation_extension/results/fashion_mnist"
mkdir -p "$RESULTS_DIR"

# List of hypothesis scripts to run (H107 to H156)
HYPOTHESES=(
    "h107_targeted_fgsm.py"
    "h108_pgd_restarts.py"
    "h109_mim.py"
    "h110_dim.py"
    "h111_eot.py"
    "h112_hsv_attack.py"
    "h113_simba.py"
    "h114_zoo.py"
    "h115_nes.py"
    "h116_sign_opt.py"
    "h117_ibp_bound.py"
    "h118_cure.py"
    "h119_alp.py"
    "h120_awp.py"
    "h121_cutout.py"
    "h122_random_erasing.py"
    "h123_augmix.py"
    "h124_manifold_mixup.py"
    "h125_lrp.py"
    "h126_smoothgrad.py"
    "h127_integrated_gradients.py"
    "h128_gradcam.py"
    "h129_forgetting_events.py"
    "h130_cscore.py"
    "h131_memorization.py"
    "h132_snapshot_ensemble.py"
    "h133_stochastic_depth.py"
    "h134_bnn_variance.py"
    "h135_uncertainty_decomp.py"
    "h136_class_margin_stats.py"
    "h137_pixel_sign_agreement.py"
    "h138_layer_ablation.py"
    "h139_vit_vs_cnn.py"
    "h140_transferability_matrix.py"
    "h141_sgd_noise.py"
    "h142_rfnn_baseline.py"
    "h143_confusion_graph.py"
    "h144_sat_adv_training.py"
    "h145_friendly_at.py"
    "h146_adv_distillation.py"
    "h147_margin_weighted_at.py"
    "h148_adv_example_overlap.py"
    "h149_saliency_spatial_pattern.py"
    "h150_confusion_ratio.py"
    "h151_gradient_phase.py"
    "h152_nearest_other_class_image.py"
    "h153_bn_sensitivity.py"
    "h154_representation_norm.py"
    "h155_meta_detector.py"
    "h156_multi_attack_metamodel.py"
    "h157_training_trajectory.py"
    "h158_confirmatory_examples.py"
    "h159_post_at_misclassification.py"
    "h160_grokking_hard_dataset.py"
    "h161_checkpoint_gradient_consistency.py"
    "h162_class_margin_shift_at.py"
)


echo "=== Starting Hypothesis Runner Loop ==="
echo "Results will be saved in $RESULTS_DIR and pushed to GitHub after each run."

for script in "${HYPOTHESES[@]}"; do
    script_path="dissertation_extension/hypotheses/$script"
    
    if [ ! -f "$script_path" ]; then
        echo "Warning: Script $script_path not found, skipping."
        continue
    fi
    
    base_name=$(basename "$script" .py)
    out_file="$RESULTS_DIR/${base_name}_output.txt"
    
    # Check if this script has already been run and pushed to avoid redundant execution
    if [ -f "$out_file" ]; then
        echo "Skipping $script (already run, output exists)"
        continue
    fi
    
    echo "--------------------------------------------------"
    echo "Running $script -> $out_file..."
    echo "Started at: $(date)"
    
    # Run the python script using the virtual environment
    if .venv/bin/python "$script_path" > "$out_file" 2>&1; then
        echo "Successfully finished $script!"
        
        # Git stage, commit, and push automatically
        git add "$out_file"
        
        # If the script generated output directories, stage them too
        output_dir="dissertation_extension/hypotheses/${base_name}_outputs"
        if [ -d "$output_dir" ]; then
            git add "$output_dir"
        fi
        
        git commit -m "run: results for $base_name"
        git push origin dissertation-extension
        echo "Pushed results for $base_name to GitHub!"
    else
        echo "Error: $script failed! Storing error log in $out_file."
        # Commit the error log anyway so you can inspect what failed
        git add "$out_file"
        git commit -m "run: error results for $base_name"
        git push origin dissertation-extension
    fi
done

echo "=== Hypothesis Runner Loop Complete ==="
