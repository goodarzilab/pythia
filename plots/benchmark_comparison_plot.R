require(ggplot2)
require(precrec)
require(cowplot)
require(parallel)
require(ROCR)
require(cvAUC)
require(RColorBrewer)
require(Hmisc)
require(dplyr)
require(tidyr)

# Usage: Rscript benchmark_comparison_plot.R <indir> <outdir> <rnafm-dir>
#
# <indir>     Root directory with one sub-directory per RBP, each containing
#             this benchmark's baseline model outputs:
#               {indir}/{rbp}/Pythia/{rbp}_validationPredictions.tsv
#                 (written by examples/run_train_rbp.sh + infer.py)
#               {indir}/{rbp}/DeepSEA/DeepSEA_Sequences_binding_and_prediction.tsv.gz
#               {indir}/{rbp}/DNABERT/DNABERT_Sequences_binding_and_prediction.tsv.gz
#               {indir}/{rbp}/DeepBind/DeepBind_Sequences_binding_and_prediction.tsv.gz
#               {indir}/{rbp}/graphProtModel/model_prediction.predictions
#               {indir}/{rbp}/PRIESTESS2/PRIESSTESS_output/test_Validation/test_posteriorProbs_PRIESSTESS_model.tsv
#             These baseline outputs are produced by their respective
#             third-party tools and are not redistributed with this repo --
#             bring your own in the formats above, or comment out the
#             baselines you do not have.
# <outdir>    Output directory for the comparison plot.
# <rnafm-dir> Root directory of RNA foundation-model linear-probing outputs
#             (RNA-FM, RNA-MSM, UTR-LM, RiNALMo, RNAErnie, ...), one
#             sub-directory per model containing a raw_scores_csv_val*
#             directory of per-RBP {rbp}_*.csv prediction files (columns:
#             label, proba). Also not redistributed with this repo.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
    stop("Usage: Rscript benchmark_comparison_plot.R <indir> <outdir> <rnafm-dir>")
}

indir <- args[1]
outdir <- args[2]
rnafm_dir <- args[3]

rna_fm_names <- c(
    "RiNALMo/RiNALMo-micro",
    "RNAErnie",
    "RNA-FM",
    "RNA-MSM",
    "UTR-LM-MRL",
    "UTR-LM-TE_EL"
)

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

missing_files <- c()


create_plot <- function(P, path, width, height, make_pdf = TRUE) {
    png(sprintf("%s.png", path), res = 200, width = width * 200, height = height * 200)
    plot(P)
    dev.off()
    if (make_pdf) {
        pdf(sprintf("%s.pdf", path), width = width, height = height)
        plot(P)
        dev.off()
    }
}


get_scalar_metrics <- function(probs, labels) {
    ord <- order(probs, decreasing = TRUE)
    probs <- probs[ord]
    labels <- labels[ord]

    n_pos <- sum(labels)
    n_neg <- sum(!labels)
    TP <- cumsum(labels)
    FP <- cumsum(!labels)
    TN <- n_neg - FP
    FN <- n_pos - TP

    TPR <- TP / n_pos
    FPR <- FP / n_neg
    Precision <- TP / (TP + FP)
    Precision[is.na(Precision)] <- 1

    MCC <- (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC[is.na(MCC)] <- 0
    best_mcc <- max(MCC, na.rm = TRUE)

    idx_90 <- max(which(FPR <= 0.1))
    if (is.infinite(idx_90)) idx_90 <- 1

    sens_90 <- TPR[idx_90]
    ppv_90 <- Precision[idx_90]

    p_idx <- which(FPR <= 0.1)
    x_p <- c(0, FPR[p_idx])
    y_p <- c(0, TPR[p_idx])

    if (max(x_p) < 0.1 && length(probs) > max(p_idx)) {
        next_idx <- max(p_idx) + 1
        x0 <- FPR[max(p_idx)]; x1 <- FPR[next_idx]
        y0 <- TPR[max(p_idx)]; y1 <- TPR[next_idx]
        y_interp <- y0 + (y1 - y0) * ((0.1 - x0) / (x1 - x0))
        x_p <- c(x_p, 0.1)
        y_p <- c(y_p, y_interp)
    }

    pAUC <- sum(diff(x_p) * (head(y_p, -1) + tail(y_p, -1)) / 2)

    return(data.frame(
        BestMCC = best_mcc,
        SensAt90Spec = sens_90,
        PPVAt90Spec = ppv_90,
        pAUC0.1 = pAUC
    ))
}


get_auc_intervals <- function(post_df, variable) {
    post_df$Chrom <- sapply(post_df$SeqNames, function(x) {
        return(unlist(strsplit(x, ":", TRUE))[2])
    })

    list.stats <- lapply(unique(post_df$Chrom), function(chrom) {
        tempdf <- post_df[post_df$Chrom == chrom, ]

        if (length(unique(tempdf$Label)) < 2) {
            warning(paste("Skipping Chromosome", chrom, "- contains only one class."))
            return(NULL)
        }

        pos_indices <- which(tempdf$Label == 1)
        neg_indices <- which(tempdf$Label == 0)

        list.local <- lapply(1:10, function(n) {
            samp_pos <- sample(pos_indices, length(pos_indices), replace = TRUE)
            samp_neg <- sample(neg_indices, length(neg_indices), replace = TRUE)
            tempdf2 <- tempdf[c(samp_pos, samp_neg), ]

            tryCatch({
                sscurves <- evalmod(scores = tempdf2[, variable], labels = tempdf2$Label)
                auc_df <- as.data.frame(auc(sscurves))
                res_df <- data.frame(
                    auROC = auc_df$aucs[auc_df$curvetypes == "ROC"],
                    auPR = auc_df$aucs[auc_df$curvetypes == "PRC"]
                )
                scalar_metrics <- get_scalar_metrics(tempdf2[, variable], tempdf2$Label)
                res_df <- cbind(res_df, scalar_metrics)
                res_df$Trial <- n
                res_df$Chrom <- chrom
                return(res_df)
            }, error = function(e) { return(NULL) })
        })

        list.local <- list.local[!sapply(list.local, is.null)]

        if (length(list.local) > 0) {
            statdf <- do.call("rbind", list.local)
            return(statdf)
        } else {
            return(NULL)
        }
    })

    list.stats <- list.stats[!sapply(list.stats, is.null)]
    statdf <- do.call("rbind", list.stats)

    metrics <- c("auROC", "auPR", "BestMCC", "SensAt90Spec", "PPVAt90Spec", "pAUC0.1")
    out_list <- list()

    if (!is.null(statdf) && nrow(statdf) > 0) {
        for (m in metrics) {
            if (m %in% names(statdf)) {
                out_list[[paste0(m, ".Mean")]] <- mean(statdf[[m]], na.rm = TRUE)
                out_list[[paste0(m, ".Upper")]] <- quantile(statdf[[m]], 0.95, na.rm = TRUE)
                out_list[[paste0(m, ".Lower")]] <- quantile(statdf[[m]], 0.05, na.rm = TRUE)
            }
        }
    } else {
        warning("No valid bootstrap iterations computed.")
        return(NULL)
    }

    return(as.data.frame(out_list))
}


get_aucdf <- function(post_df, variable) {
    if (length(unique(post_df$Label)) < 2) {
        warning(sprintf("Skipping %s for %s: only one class present.", variable, post_df$Sample[1]))
        return(NULL)
    }
    sscurves <- tryCatch(
        evalmod(scores = post_df[, variable], labels = post_df$Label),
        error = function(e) {
            warning(sprintf("evalmod failed for %s: %s", post_df$Sample[1], e$message))
            return(NULL)
        }
    )
    if (is.null(sscurves)) return(NULL)
    auc_df <- auc(sscurves)
    rownames(auc_df) <- auc_df$curvetypes

    scalar_metrics <- get_scalar_metrics(post_df[, variable], post_df$Label)

    post_df$Chrom <- sapply(post_df$SeqNames, function(x) {
        return(unlist(strsplit(x, ":", TRUE))[2])
    })
    cv_auc_df <- get_auc_intervals(post_df, variable)

    ad_df <- data.frame(
        Sample = post_df$Sample[1], Experiment = variable,
        X = signif(c(sscurves$rocs[[1]]$x, sscurves$prcs[[1]]$x), 3),
        Y = signif(c(sscurves$rocs[[1]]$y, sscurves$prcs[[1]]$y), 3),
        Method = c(
            rep("auROC", length(sscurves$rocs[[1]]$x)),
            rep("auPRC", length(sscurves$prcs[[1]]$x))
        ),
        AUC = c(
            rep(auc_df["ROC", "aucs"], length(sscurves$rocs[[1]]$x)),
            rep(auc_df["PRC", "aucs"], length(sscurves$prcs[[1]]$x))
        )
    )

    for (m in colnames(scalar_metrics)) {
        ad_df[[m]] <- scalar_metrics[[m]]
    }

    for (varname in colnames(cv_auc_df)) {
        ad_df[, varname] <- cv_auc_df[, varname]
    }

    return(ad_df)
}


rbps <- dir(indir)
rbps <- rbps[!grepl("(-myc|_motif|pseudoknot)", rbps, ignore.case = TRUE)]


aucdf <- do.call("rbind", lapply(rbps, function(rbp) {
    message(sprintf("\n=== RBP: %s ===", rbp))
    paths <- c(
        sprintf("%s/%s/Pythia/%s_validationPredictions.tsv", indir, rbp, rbp),
        sprintf("%s/%s/DeepSEA/DeepSEA_Sequences_binding_and_prediction.tsv.gz", indir, rbp),
        sprintf("%s/%s/DNABERT/DNABERT_Sequences_binding_and_prediction.tsv.gz", indir, rbp),
        sprintf("%s/%s/DeepBind/DeepBind_Sequences_binding_and_prediction.tsv.gz", indir, rbp),
        sprintf("%s/%s/graphProtModel/model_prediction.predictions", indir, rbp),
        sprintf(
            "%s/%s/PRIESTESS2/PRIESSTESS_output/test_Validation/test_posteriorProbs_PRIESSTESS_model.tsv",
            indir, rbp
        )
    )
    adnames <- c("Pythia", "DeepSEA", "DNABERT", "DeepBind", "GraphProt", "PRIESSTESS")

    for (rna_fm_name in rna_fm_names) {
        rna_fm_dir_ <- sprintf("%s/%s", rnafm_dir, rna_fm_name)
        score_dir <- dir(rna_fm_dir_)[grep("raw_scores_csv_val", dir(rna_fm_dir_))][1]
        rna_fm_dir_full <- sprintf("%s/%s/%s", rnafm_dir, rna_fm_name, score_dir)
        files_in_dir <- dir(rna_fm_dir_full, full.names = TRUE)

        matched_file <- grep(sprintf("^%s_", rbp), basename(files_in_dir), value = TRUE)
        if (length(matched_file) > 0) {
            rna_fm_file <- file.path(rna_fm_dir_full, matched_file[1])
            paths <- c(paths, rna_fm_file)
            adnames <- c(adnames, rna_fm_name)
        } else {
            missing_files <<- c(missing_files, sprintf("%s/%s", rbp, rna_fm_name))
            warning(sprintf("No RNA foundation model file found for %s in %s", rbp, rna_fm_dir_full))
        }
    }

    if (!file.exists(paths[1])) {
        warning(sprintf("Pythia predictions not found for %s — skipping.", rbp))
        return(NULL)
    }

    list.out <- list()
    ref_positions <- NULL

    for (i in 1:length(adnames)) {
        inpath <- paths[i]
        if (file.exists(inpath)) {
            adname <- adnames[i]
            indf <- read.csv(inpath, header = TRUE, sep = "\t")
            if (i == 1) {
                # Pythia: score column is pred_prob_bound, map to Pred.Response
                ref_positions <- indf$SeqNames
                indf$Pred.Response <- indf$pred_prob_bound
            } else if (i == 6) {
                colnames(indf) <- c("SeqNames", "Response", "Pred.Response")
                indf$Response <- ifelse(indf$Response, "Bound", "Unbound")
                indf$SeqNames[indf$Response == "Bound"] <- sample(
                    ref_positions, sum(indf$Response == "Bound"), replace = TRUE
                )
                indf$SeqNames[indf$Response != "Bound"] <- sample(
                    ref_positions, sum(indf$Response != "Bound"), replace = TRUE
                )
            } else if (i >= 7) {
                indf <- read.csv(inpath, header = TRUE, sep = ",")
                indf$Response <- ifelse(indf$label == 1, "Bound", "Unbound")
                indf$Pred.Response <- indf$proba
            } else {
                if ("SeqNames" %in% colnames(indf) && !is.null(ref_positions)) {
                    indf <- indf[indf$SeqNames %in% ref_positions, ]
                }
            }
            indf$RBP <- rbp

            if (adname == "GraphProt") {
                indf <- read.csv(inpath, header = FALSE, sep = "\t")
                colnames(indf) <- c("Sequence", "Prediction.Label", "Posterior")
                indf <- indf[indf$Sequence %in% ref_positions, ]
                if (nrow(indf) == 0) {
                    warning(sprintf("GraphProt: no matching sequences for %s after filtering — skipping.", rbp))
                    next
                }
                indf$Bound <- ifelse(grepl("Shuff", indf$Sequence), FALSE, TRUE)
                tempdf <- data.frame(
                    Sample = rbp, Model = adname,
                    Posterior = indf$Posterior, Label = indf$Bound,
                    SeqNames = indf$Sequence
                )
            } else {
                seq_names_col <- if (!is.null(indf$SeqNames) && length(indf$SeqNames) > 0) {
                    indf$SeqNames
                } else {
                    sample(ref_positions, nrow(indf), replace = TRUE)
                }
                # Handle Response as "Bound"/"Unbound" strings OR as 0/1 numeric
                label_col <- if (is.character(indf$Response) || is.factor(indf$Response)) {
                    indf$Response == "Bound"
                } else {
                    as.logical(indf$Response)
                }
                tempdf <- data.frame(
                    Sample = rbp, Model = adname,
                    Posterior = indf$Pred.Response, Label = label_col,
                    SeqNames = seq_names_col
                )
            }
            aucdf_rbp <- tryCatch({
                res <- get_aucdf(tempdf, "Posterior")
                if (!is.null(res)) {
                    res$Group <- rbp
                    res$RBP <- rbp
                    res$Model <- adname
                    roc_auc <- signif(unique(res$AUC[res$Method == "auROC"]), 4)
                    prc_auc <- signif(unique(res$AUC[res$Method == "auPRC"]), 4)
                    message(sprintf("  %-20s  auROC=%.4f  auPRC=%.4f", adname, roc_auc, prc_auc))
                }
                res
            }, error = function(e) {
                warning(sprintf("get_aucdf failed for %s/%s: %s", rbp, adname, e$message))
                NULL
            })
            if (!is.null(aucdf_rbp)) {
                list.out <- c(list.out, list(aucdf_rbp))
            }
        }
    }

    if (length(list.out) > 0) {
        aucdf_rbp <- do.call("rbind", list.out)
        aucdf_rbp$Experiment <- aucdf_rbp$Model
        aucdf_rbp$Group <- paste(aucdf_rbp$RBP, aucdf_rbp$Model, aucdf_rbp$Method)
        textdf <- unique(aucdf_rbp[, setdiff(colnames(aucdf_rbp), c("X", "Y"))])
        textdf$Label <- paste(textdf$Method, signif(textdf$AUC, 3))
        textdf <- textdf[order(textdf$AUC), ]
        textdf <- textdf[order(textdf$Method), ]
        textdf$X <- 0.75
        textdf$Y <- rep(seq(0.1, 0.7, length.out = nrow(textdf) / 2), 2)
        return(textdf)
    } else {
        return(NULL)
    }
}))


saveRDS(aucdf, file = sprintf("%s/auc_df_subset.RDS", outdir), compress = TRUE)
aucdf <- readRDS(sprintf("%s/auc_df_subset.RDS", outdir))

aucdf[, "Experiment"] <- gsub("RiNALMo/RiNALMo-micro", "RiNALMo", aucdf[, "Experiment"])
aucdf <- aucdf[aucdf$Experiment != "DNABERT", ]

non_fm_models <- c("Pythia", "DeepSEA", "DeepBind", "GraphProt", "PRIESSTESS")
fm_models <- c(
    "Pythia", "RNAErnie", "RNA-MSM",
    "UTR-LM-MRL", "UTR-LM-TE_EL", "RNA-FM", "RiNALMo"
)

all_models <- unique(c(non_fm_models, fm_models))

my_palette <- c(
    brewer.pal(9, "Set1"),
    brewer.pal(8, "Set2"),
    brewer.pal(8, "Set3")
)[1:length(all_models)]
my_palette[6] <- "brown"
names(my_palette) <- all_models

freq_rbp <- table(aucdf$RBP)
rbps <- names(freq_rbp)

aucdf <- aucdf[aucdf$RBP %in% rbps, ]

order_rbps <- sort(sapply(rbps, function(x) {
    tempdf <- aucdf[aucdf$Sample == x & aucdf$Experiment == "Pythia" & aucdf$Method == "auPRC", ]
    if (nrow(tempdf) == 0) return(NA)
    return(tempdf$AUC[1])
}))
order_rbps <- order_rbps[!is.na(order_rbps)]

aucdf$Sample <- factor(aucdf$Sample, levels = names(order_rbps))


plot_data_1 <- aucdf[aucdf$Method %in% c("auROC", "auPRC") & aucdf$Experiment %in% non_fm_models, ]

rbp_freqs <- table(plot_data_1$Sample)
select_rbps <- names(rbp_freqs[rbp_freqs == max(rbp_freqs)])
num_rbps <- length(select_rbps)
plot_data_1 <- plot_data_1[plot_data_1$Sample %in% select_rbps, ]

p1_means <- aggregate(AUC ~ Experiment, data = plot_data_1[plot_data_1$Method == "auPRC", ], mean)
p1_order <- p1_means$Experiment[order(p1_means$AUC, decreasing = TRUE)]
plot_data_1$Experiment <- factor(plot_data_1$Experiment, levels = p1_order)
axis_colors_1 <- my_palette[levels(plot_data_1$Experiment)]

P1 <- ggplot(
    plot_data_1,
    aes(x = Experiment, y = AUC, fill = Experiment, alpha = Method)
) +
    stat_summary(
        fun = "mean", geom = "bar",
        position = position_dodge(width = 0.8), width = 0.7
    ) +
    stat_summary(
        aes(group = Method), fun.data = mean_cl_normal, geom = "errorbar",
        position = position_dodge(width = 0.8), width = 0.2, color = "black", alpha = 1
    ) +
    theme_bw(base_size = 14) +
    xlab("") +
    ylab("Area Under Curve (AUC)") +
    scale_y_continuous(limits = c(0, 1)) +
    scale_fill_manual(name = "Experiment", values = my_palette) +
    scale_alpha_manual(values = c("auROC" = 0.4, "auPRC" = 0.9)) +
    geom_hline(yintercept = 0.5, linetype = 2, color = "gray") +
    geom_hline(yintercept = 0.1, linetype = 3, color = "darkgray") +
    ggtitle(sprintf("Supervised models (n = %d)", num_rbps)) +
    theme(
        legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1, color = axis_colors_1, face = "bold"),
        axis.title.y = element_text(face = "bold"),
        plot.title = element_text(face = "bold")
    )


plot_data_2 <- aucdf[aucdf$Method %in% c("auROC", "auPRC") & aucdf$Experiment %in% fm_models, ]

rbp_freqs_2 <- table(plot_data_2$Sample)
select_rbps_2 <- names(rbp_freqs_2[rbp_freqs_2 == max(rbp_freqs_2)])
num_rbps_2 <- length(select_rbps_2)
plot_data_2 <- plot_data_2[plot_data_2$Sample %in% select_rbps_2, ]

p2_means <- aggregate(AUC ~ Experiment, data = plot_data_2[plot_data_2$Method == "auPRC", ], mean)
p2_order <- p2_means$Experiment[order(p2_means$AUC, decreasing = TRUE)]
plot_data_2$Experiment <- factor(plot_data_2$Experiment, levels = p2_order)
axis_colors_2 <- my_palette[levels(plot_data_2$Experiment)]

P2 <- ggplot(
    plot_data_2,
    aes(x = Experiment, y = AUC, fill = Experiment, alpha = Method)
) +
    stat_summary(
        fun = "mean", geom = "bar",
        position = position_dodge(width = 0.8), width = 0.7
    ) +
    stat_summary(
        aes(group = Method), fun.data = mean_cl_normal, geom = "errorbar",
        position = position_dodge(width = 0.8), width = 0.2, color = "black", alpha = 1
    ) +
    theme_bw(base_size = 14) +
    xlab("") +
    ylab("") +
    scale_y_continuous(limits = c(0, 1)) +
    scale_fill_manual(name = "Experiment", values = my_palette) +
    scale_alpha_manual(values = c("auROC" = 0.4, "auPRC" = 0.9)) +
    geom_hline(yintercept = 0.5, linetype = 2, color = "gray") +
    geom_hline(yintercept = 0.1, linetype = 3, color = "darkgray") +
    ggtitle(sprintf("Linear probing (n = %d)", num_rbps_2)) +
    theme(
        legend.position = "right",
        axis.text.x = element_text(angle = 45, hjust = 1, color = axis_colors_2, face = "bold"),
        plot.title = element_text(face = "bold")
    )

combined_plot <- plot_grid(
    P1, P2,
    labels = c("A", "B"),
    ncol = 2,
    rel_widths = c(1, 2.0),
    align = "h",
    axis = "bt"
)


valid_rbps_intersection <- intersect(select_rbps, select_rbps_2)
if (length(valid_rbps_intersection) == 0) {
    warning("No RBP found with complete data across ALL models. Using RBP with most coverage.")
    target_rbp <- names(sort(table(aucdf$Sample), decreasing = TRUE))[1]
} else {
    target_rbp <- valid_rbps_intersection[1]
}

target_rbp <- "HNRNPA0"
print(paste("Selected RBP for detailed curves:", target_rbp))


get_raw_data_for_rbp <- function(rbp, indir, rnafm_dir, rna_fm_names) {
    adnames_orig <- c("Pythia", "DeepSEA", "DNABERT", "DeepBind", "GraphProt", "PRIESSTESS")
    paths_orig <- c(
        sprintf("%s/%s/Pythia/%s_validationPredictions.tsv", indir, rbp, rbp),
        sprintf("%s/%s/DeepSEA/DeepSEA_Sequences_binding_and_prediction.tsv.gz", indir, rbp),
        sprintf("%s/%s/DNABERT/DNABERT_Sequences_binding_and_prediction.tsv.gz", indir, rbp),
        sprintf("%s/%s/DeepBind/DeepBind_Sequences_binding_and_prediction.tsv.gz", indir, rbp),
        sprintf("%s/%s/graphProtModel/model_prediction.predictions", indir, rbp),
        sprintf(
            "%s/%s/PRIESTESS2/PRIESSTESS_output/test_Validation/test_posteriorProbs_PRIESSTESS_model.tsv",
            indir, rbp
        )
    )

    raw_list <- list()

    ref_df <- read.csv(paths_orig[1], header = TRUE, sep = "\t")
    ref_positions <- ref_df$SeqNames

    for (i in 1:length(adnames_orig)) {
        path <- paths_orig[i]
        name <- adnames_orig[i]
        if (file.exists(path)) {
            indf <- read.csv(path, header = TRUE, sep = "\t")
            if (i == 1) {
                # Pythia: score column is pred_prob_bound
                temp <- data.frame(
                    Model = name,
                    Score = indf$pred_prob_bound,
                    Label = indf$Response == "Bound"
                )
            } else if (i == 5) {
                indf <- read.csv(path, header = FALSE, sep = "\t")
                colnames(indf) <- c("Sequence", "Prediction.Label", "Posterior")
                indf <- indf[indf$Sequence %in% ref_positions, ]
                temp <- data.frame(
                    Model = name,
                    Score = indf$Posterior,
                    Label = !grepl("Shuff", indf$Sequence)
                )
            } else if (i == 6) {
                colnames(indf) <- c("SeqNames", "Response", "Pred.Response")
                temp <- data.frame(
                    Model = name,
                    Score = indf$Pred.Response,
                    Label = indf$Response == 1
                )
            } else {
                if ("SeqNames" %in% colnames(indf)) {
                    indf <- indf[indf$SeqNames %in% ref_positions, ]
                }
                label_col <- if (is.character(indf$Response) || is.factor(indf$Response)) {
                    indf$Response == "Bound"
                } else {
                    as.logical(indf$Response)
                }
                temp <- data.frame(
                    Model = name,
                    Score = indf$Pred.Response,
                    Label = label_col
                )
            }
            raw_list[[name]] <- temp
        }
    }

    for (fm in rna_fm_names) {
        fm_path_root <- sprintf("%s/%s", rnafm_dir, fm)
        if (!dir.exists(fm_path_root)) next

        score_dir <- dir(fm_path_root)[grep("raw_scores_csv_val", dir(fm_path_root))][1]
        fm_full_dir <- sprintf("%s/%s", fm_path_root, score_dir)
        matched_file <- grep(sprintf("^%s_", rbp), basename(dir(fm_full_dir)), value = TRUE)

        if (length(matched_file) > 0) {
            fm_file <- file.path(fm_full_dir, matched_file[1])
            indf <- read.csv(fm_file, header = TRUE, sep = ",")
            name_clean <- gsub("RiNALMo/RiNALMo-micro", "RiNALMo", fm)
            temp <- data.frame(
                Model = name_clean,
                Score = indf$proba,
                Label = indf$label == 1
            )
            raw_list[[name_clean]] <- temp
        }
    }
    return(do.call(rbind, raw_list))
}

raw_data_top <- get_raw_data_for_rbp(target_rbp, indir, rnafm_dir, rna_fm_names)

curve_coords <- do.call(rbind, lapply(unique(raw_data_top$Model), function(m) {
    subset_data <- raw_data_top[raw_data_top$Model == m, ]
    msmm <- evalmod(scores = subset_data$Score, labels = subset_data$Label)

    aucs <- auc(msmm)
    roc_auc <- signif(aucs$aucs[aucs$curvetypes == "ROC"], 2)
    prc_auc <- signif(aucs$aucs[aucs$curvetypes == "PRC"], 2)

    legend_label <- sprintf("%s (auROC: %s, auPR: %s)", m, roc_auc, prc_auc)

    roc_obj <- msmm$rocs[[1]]
    roc <- data.frame(x = roc_obj$x, y = roc_obj$y)
    roc$Method <- "ROC"

    pr_obj <- msmm$prcs[[1]]
    pr <- data.frame(x = pr_obj$x, y = pr_obj$y)
    pr$Method <- "PR"

    df <- rbind(roc, pr)
    df$Model <- m
    df$LegendLabel <- legend_label
    return(df)
}))


plot_detailed_curves <- function(curve_df, model_subset, title, color_palette) {
    plot_df <- curve_df[curve_df$Model %in% model_subset, ]

    unique_mapping <- unique(plot_df[, c("Model", "LegendLabel")])
    legend_colors <- color_palette[unique_mapping$Model]
    names(legend_colors) <- unique_mapping$LegendLabel

    df_roc <- plot_df[plot_df$Method == "ROC", ]
    df_pr <- plot_df[plot_df$Method == "PR", ]

    p_roc <- ggplot(df_roc, aes(x = x, y = y, color = LegendLabel)) +
        geom_line(linewidth = 1) +
        theme_bw(base_size = 12) +
        geom_abline(intercept = 0, slope = 1, linetype = 2, color = "grey") +
        scale_color_manual(name = "", values = legend_colors) +
        scale_x_continuous(limits = c(0, 1)) +
        scale_y_continuous(limits = c(0, 1)) +
        xlab(expression("False positive rate: " * frac(FP, FP + TN))) +
        ylab(expression("True positive rate: " * frac(TP, TP + FN))) +
        ggtitle("ROC") +
        theme(
            legend.position = "none",
            plot.title = element_text(hjust = 0.5, face = "bold"),
            axis.title = element_text(face = "bold")
        )

    p_pr <- ggplot(df_pr, aes(x = x, y = y, color = LegendLabel)) +
        geom_line(linewidth = 1) +
        theme_bw(base_size = 12) +
        geom_hline(yintercept = 0.1, linetype = 2, color = "grey") +
        scale_color_manual(name = "", values = legend_colors) +
        scale_x_continuous(limits = c(0, 1)) +
        scale_y_continuous(limits = c(0, 1)) +
        xlab(expression("Recall: " * frac(TP, TP + FN))) +
        ylab(expression("Precision: " * frac(TP, TP + FP))) +
        ggtitle("PR") +
        theme(
            legend.position = "right",
            plot.title = element_text(hjust = 0.5, face = "bold"),
            axis.title = element_text(face = "bold")
        )

    combined <- plot_grid(p_roc, p_pr, ncol = 2, rel_widths = c(1, 1.75))
    title_theme <- ggdraw() + draw_label(title, fontface = "bold", size = 14)
    return(plot_grid(title_theme, combined, ncol = 1, rel_heights = c(0.1, 1)))
}


supervised_subset <- setdiff(non_fm_models, "DNABERT")
P_B <- plot_detailed_curves(
    curve_coords, supervised_subset,
    sprintf("Supervised: %s", target_rbp), my_palette
)

fm_subset <- fm_models
P_C <- plot_detailed_curves(
    curve_coords, fm_subset,
    sprintf("Foundation Models: %s", target_rbp), my_palette
)

final_layout <- plot_grid(
    combined_plot,
    plot_grid(P_B, P_C, labels = c("B", "C"), ncol = 1, nrow = 2, rel_heights = c(1, 1)),
    nrow = 2,
    rel_heights = c(1, 2)
)

create_plot(
    final_layout,
    sprintf("%s/rbp_benchmark_comparison", outdir),
    width = 12,
    height = 12
)
