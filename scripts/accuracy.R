library(readtext)
library(stringr)
library(ggplot2)
library(reshape)
library(wesanderson)
library(RColorBrewer)
library(tidyverse)
library(viridis)
library(dplyr)

fname_ofer15_voting = '/media/asem/store/experimental/markovian_features/performance_ofer15_voting.txt'
fname_ofer15_acc = '/media/asem/store/experimental/markovian_features/performance_ofer15_acc.txt'
# fname_nogrouping_voting = '/media/asem/store/experimental/markovian_features/nogrouping-voting.txt'
# fname_nogrouping_acc = '/media/asem/store/experimental/markovian_features/nogrouping-acc.txt'

coloursPalettes = unlist(c(
  colorRampPalette(brewer.pal(6, "Reds"))(5) ,
  colorRampPalette(brewer.pal(6, "Purples"))(5) ,
  colorRampPalette(brewer.pal(6, "Oranges"))(5) ,
  colorRampPalette(brewer.pal(6, "Greens"))(5),
  colorRampPalette(brewer.pal(6, "Blues"))(5)
))

MAX_ORDER = 6
MIN_ORDER = 2

coloursPalettesIdx = function(orders) {
  mino = 0
  maxo = 0
  tokens = unlist(strsplit(as.character(orders), "-"))
  mino = MAX_ORDER - as.numeric(tokens[1]) 
  maxo = MAX_ORDER - as.numeric(tokens[2]) 
  return((MAX_ORDER - MIN_ORDER) * maxo + mino  + 1)
}

coloursPalettesIdxSwapped = function(orders) {
  mino = 0
  maxo = 0
  tokens = unlist(strsplit(as.character(orders), "-"))
  mino = MAX_ORDER - as.numeric(tokens[1]) 
  maxo = MAX_ORDER - as.numeric(tokens[2]) 
  # print(orders)
  # print((MAX_ORDER - MIN_ORDER + 1) * maxo + mino + 1)
  return((MAX_ORDER - MIN_ORDER + 1) * maxo + mino  + 1)
}


coloursPalettesIndices = function(ranges)
  lapply(ranges, coloursPalettesIdx)
coloursPalettesIndicesSwapped = function(ranges)
  lapply(ranges, coloursPalettesIdxSwapped)

labels = function(orders) {
  mino = 0
  maxo = 0
  tokens = unlist(strsplit(as.character(orders), "-"))
  return(sprintf("(%s-%s)", tokens[1] , tokens[2]))
}

labels = function(orders) {
  mino = 0
  maxo = 0
  tokens = unlist(strsplit(as.character(orders), "-"))
  return(sprintf("(%s-%s)", tokens[2] , tokens[1]))
  
}


reorder_within <-
  function(x,
           by,
           within,
           fun = mean,
           sep = "___",
           ...) {
    new_x <- paste(x, within, sep = sep)
    stats::reorder(new_x, by, FUN = fun)
  }


#' @rdname reorder_within
#' @export
scale_x_reordered <- function(..., sep = "___") {
  reg <- paste0(sep, ".+$")
  ggplot2::scale_x_discrete(
    labels = function(x)
      gsub(reg, "", x),
    ...
  )
}


#' @rdname reorder_within
#' @export
scale_y_reordered <- function(..., sep = "___") {
  reg <- paste0(sep, ".+$")
  ggplot2::scale_y_discrete(
    labels = function(x)
      gsub(reg, "", x),
    ...
  )
}


metrics = c(
  'Cosine',
  'Kullback-Leibler',
  'Chi-squared',
  'DPD (α=1)',
  'DPD (α=2)',
  'DPD (α=3)',
  'Gaussian RBF' ,
  'Bhattacharyya',
  'Hellinger'
)


names(metrics) = c('cos' ,
                   'kl' ,
                   'chi' ,
                   'dpd1',
                   'dpd2' ,
                   'dpd3',
                   'gaussian'  ,
                   'bhat' ,
                   'hell')


extract_results = function(fname) {
  txt = readtext(fname)$text
  txt = unlist(strsplit(txt , "[Params]", fixed = TRUE))
  txt = txt[-1]
  df <- data.frame(txt)
  params = regmatches(df$txt, gregexpr("\\[.+?\\]", df$txt))
  order_ = lapply(params , function(item) {
    token = gsub("\\[|\\]", "", item[1])
    token = unlist(strsplit(token, ":"))[2]
    return(gsub("\\(|\\)", "", token))
  })
  order_swapped_ = lapply(params , function(item) {
    token = gsub("\\[|\\]", "", item[1])
    token = unlist(strsplit(token, ":"))[2]
    token = gsub("\\(|\\)", "", token)
    return(sprintf("%s-%s" , unlist(strsplit(token, "-"))[2], unlist(strsplit(token, "-"))[1]))
  })
  range_magnitude_ = lapply(params , function(item) {
    token = gsub("\\[|\\]", "", item[1])
    token = unlist(strsplit(token, ":"))[2]
    return(as.numeric(as.numeric(unlist(
      str_split(token, "-")
    )[2]) - as.numeric(unlist(
      str_split(token, "-")
    )[1])) + 1 )
  })
  minorder_ = lapply(params , function(item) {
    token = gsub("\\[|\\]", "", item[1])
    token = unlist(strsplit(token, ":"))[2]
    token = unlist(str_split(token, "-"))[1]
    return(as.numeric(token))
  })
  maxorder_ = lapply(params , function(item) {
    token = gsub("\\[|\\]", "", item[1])
    token = unlist(strsplit(token, ":"))[2]
    token = unlist(str_split(token, "-"))[2]
    return(as.numeric(token))
  })
  metric_ = lapply(params , function(item) {
    token = gsub("\\[|\\]", "", item[2])
    token = unlist(strsplit(token, ":"))[2]
    return(metrics[token])
  })
  accuracy_ = lapply(df$txt , function(item) {
    tokens = unlist(strsplit(as.character(item), "\n", fixed = TRUE))
    performance_line = str_trim(tokens[grepl("Overall Accuracy", tokens, fixed =
                                               TRUE)])
    value = unlist(strsplit(performance_line[1], ":"))
    accuracy = unlist(strsplit(value[2] , " "))[1]
    return(as.numeric(accuracy))
  })
  accuracy_std_ = lapply(df$txt , function(item) {
    tokens = unlist(strsplit(as.character(item), "\n", fixed = TRUE))
    performance_line = str_trim(tokens[grepl("Overall Accuracy", tokens, fixed =
                                               TRUE)])
    value = unlist(strsplit(performance_line[1], ":"))[2]
    std = unlist(gsub("\\(|\\)", "", unlist(strsplit(value , " "))[3]))
    return(as.numeric(substring(std, 2)))
  })
  
  mcc_ = lapply(df$txt , function(item) {
    tokens = unlist(strsplit(as.character(item), "\n", fixed = TRUE))
    performance_line = str_trim(tokens[grepl("MCC", tokens, fixed =
                                               TRUE)])
    value = unlist(strsplit(performance_line[1], ":"))
    accuracy = unlist(strsplit(value[2] , " "))[1]
    return(as.numeric(accuracy))
  })
  
  mcc_std_ = lapply(df$txt , function(item) {
    tokens = unlist(strsplit(as.character(item), "\n", fixed = TRUE))
    performance_line = str_trim(tokens[grepl("MCC", tokens, fixed =
                                               TRUE)])
    value = unlist(strsplit(performance_line[1], ":"))[2]
    std = unlist(gsub("\\(|\\)", "", unlist(strsplit(value , " "))[3]))
    return(as.numeric(substring(std, 2)))
  })
  
  results = data.frame(
    orders_range = unlist(order_) ,
    orders_range_swapped = unlist(order_swapped_),
    range_magnitude = unlist(range_magnitude_),
    min_order = unlist(minorder_) ,
    max_order = unlist(maxorder_),
    metric = unlist(metric_) ,
    overall_accuracy = unlist(accuracy_),
    overall_accuracy_std = unlist(accuracy_std_),
    mcc = unlist(mcc_),
    mcc_std = unlist(mcc_std_)
  )
  return(results)
}


plot_performance_vs_ordersrange = function(data) {
  data_filtered = filter(
    data,
    metric %in% c(
      'Bhattacharyya' ,
      'Cosine',
      'DPD (α=1)',
      'Gaussian RBF' ,
      'Hellinger',
      'Kullback-Leibler'
    )
  )
  data_sorted = arrange(data_filtered , metric, -max_order , range_magnitude )
  max_order_palette = values=wes_palette(n=5, name="BottleRocket2") 
  names( max_order_palette ) = levels( data_sorted$max_order )
  range_label = unlist(lapply( data_sorted$orders_range_swapped , function(range){
    tokens = unlist(strsplit( as.character(range) , "-" , fixed = TRUE ))

    if( tokens[1] == tokens[2] ) {
      return(tokens[1])
      }
    else {
      return(sprintf("(%s-%s)", tokens[1] , tokens[2]))
    }
  }))
  names(range_label) = data_sorted$orders_range_swapped

  ggplot(
    data_sorted,
    aes(
      y = overall_accuracy,
      x =   reorder_within(orders_range_swapped, -max_order, range_magnitude) ,
      fill =  as.factor(max_order)  ,
      group = as.factor(max_order)
      # color = as.factor(max_order)
    )
  ) +
    scale_x_reordered( ) +
    geom_bar(stat = "identity") +
    facet_wrap(~metric, scales = "free_x") +
    coord_cartesian(ylim = c(0.5, 0.8)) +
    scale_fill_manual( values =  max_order_palette ) +
    # scale_color_manual( values =  max_order_palette ) +
    theme(
      legend.position = "none",
      axis.title.x = element_text(size=14,),
      axis.title.y = element_text(size=14),
      axis.text.x = element_text(size = 12, 
                                 angle = 45,
                                 colour =  max_order_palette[ as.factor(data_sorted$max_order) ] ),
      axis.ticks.x = element_blank()
    ) +
      xlab("Orders Range") +
      ylab("Overall Accuracy (10-fold, stratified)")+
    geom_errorbar(
      aes(
        ymin = overall_accuracy - overall_accuracy_std,
        ymax = overall_accuracy + overall_accuracy_std
      ),
      size = .3,
      # Thinner lines
      width = .2,
      position = position_dodge(.9)
    ) 
}

plot_data = function( data ){
    # ggplot(ofer15_voting, aes(y=overall_accuracy, x=metric, fill=metric)) +
    #   geom_bar( stat="identity") +
    #   facet_wrap(~orders_range) +
    #   coord_cartesian(ylim=c(0.5,0.8) ) +
    #   scale_fill_brewer(palette="Paired")+
    #   theme(axis.title.x=element_blank(),
    #         axis.text.x=element_blank(),
    #         axis.ticks.x=element_blank()) +
    #   geom_errorbar(aes(ymin=overall_accuracy-overall_accuracy_std, ymax=overall_accuracy+overall_accuracy_std),
    #                 size=.3,    # Thinner lines
    #                 width=.2,
    #                 position=position_dodge(.9))


  
  # pd <- position_dodge(0.1) # move them .05 to the left and right
  # data_lines = filter(data, metric %in% c('Cosine',
  #                                         'Kullback-Leibler',
  #                                         'DPD (α=1)',
  #                                         'Gaussian RBF' ,
  #                                         'Bhattacharyya' ))
  # ggplot(data_lines,
  #        aes(
  #          shape=metric,
  #          x = orders_range,
  #          y = overall_accuracy,
  #          group = metric,
  #          color=metric,
  #          width = 0.05
  #        )) +
  #     xlab("Orders Range") +
  #     ylab("Overall Accuracy (10-fold, stratified)")+
  #   geom_errorbar(
  #     aes(
  #       ymin = overall_accuracy - overall_accuracy_std,
  #       ymax = overall_accuracy + overall_accuracy_std
  #     ),
  #     width = .1,
  #     position = pd
  #   ) +
  #   geom_point( position = pd)
  
  # bardata = filter(data, metric %in% c('Cosine',
  #                                      'DPD (α=1)',
  #                                      'Gaussian RBF' ))
  # ggplot(bardata, aes(x = reorder_within(orders_range_swapped,-max_order,-min_order),
  #                     y = overall_accuracy , fill = as.factor(max_order) )) +
  #   scale_x_reordered() +
  #   geom_bar(position = position_dodge(), stat = "identity") +
  #   geom_errorbar(
  #     aes(
  #       ymin = overall_accuracy - overall_accuracy_std,
  #       ymax = overall_accuracy + overall_accuracy_std
  #     ),
  #     width = .2,
  #     # Width of the error bars
  #     position = position_dodge(.9)
  #   ) +
  #   coord_cartesian(ylim = c(0.5, 0.8)) +
  #   scale_fill_manual(values=wes_palette(n=5, name="Rushmore")) +
  #   theme(legend.position="none") +
  #   xlab("Orders Range") +
  #   ylab("Overall Accuracy (10-fold, stratified)")+
  #   facet_wrap(~ metric , nrow=3, scales = "free_x")
}

plot_voting_vs_acc_MCC = function( voting , acc , selected_metric = c("Cosine"))
{
  voting = filter( voting,  metric %in% selected_metric )
  acc = filter( acc,  metric %in% selected_metric )
  voting$scheme = rep("MacroSimilarityVoting", nrow(voting))
  acc$scheme = rep("MacroSimilarityAccumulative Similarity", nrow(acc))
  data = rbind(voting,acc) 
  max_order_palette = wes_palette(n=2, name="BottleRocket2") 
  names( max_order_palette ) = levels( data$max_order )
  
  # data = arrange( data , desc(max_order), range_magnitude )
  data <- data[with(data, order(-range_magnitude, min_order)),]
  data$orders_range <- factor(data$orders_range, levels = unique(data$orders_range))
  ggplot(
    data  ,
    aes(
      y = mcc,
      x = orders_range
    )
  ) +
    facet_grid(. ~ desc(range_magnitude), scales="free_x", space="free_x" , switch = "x" ,
               labeller= function(labels) lapply(labels , function(label) -as.numeric(label))) +
    geom_bar(aes(fill =  scheme),position="dodge",stat = "identity") +
    geom_errorbar(
      position=position_dodge(0.9),
      aes(
        group = as.factor(scheme),
        ymin = mcc - mcc_std,
        ymax = mcc + mcc_std
      ),
      size = .3,
      # Thinner lines
      width = .2 
    )+
    theme_classic() +
    coord_cartesian(ylim = c(0.5, 0.75)) +
    theme(
      axis.title.x = element_text(size=14,),
      axis.title.y = element_text(size=14),
      axis.text.x = element_text(size = 14,
                                 angle = 30 ),
      axis.ticks.x = element_blank()
    )  + guides(fill=guide_legend(title="Classification Scheme")) +
    xlab("Orders Range (Grouped by Range Size)") +
    ylab("MCC (10-fold, stratified)")
    
}

ofer15_voting  = extract_results(fname_ofer15_voting)
ofer15_acc  = extract_results(fname_ofer15_acc)
# nogrouping_voting  = extract_results(fname_nogrouping_voting)
# nogrouping_acc = extract_results(fname_ofer15_acc)

# plot_performance_vs_ordersrange(ofer15_voting)
plot_performance_vs_ordersrange(ofer15_acc)
# plot_performance_vs_ordersrange(nogrouping_voting)
# plot_performance_vs_ordersrange(nogrouping_acc)


# plot_voting_vs_acc_MCC(ofer15_voting,ofer15_acc)

