
package com.namsor.oss.classify.bayes;

import java.io.StringWriter;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Explain the details of the Naive Bayes Classification, ie. formulae and algebraic
 * expression. This will re-run the algorithm but append additional information :
 * - likelyhood values
 * - likelyhood formulae
 * - likelyhood expressions
 *
 * @author elian
 */
public class NaiveBayesExplainerImpl extends AbstractNaiveBayesImpl implements INaiveBayesExplainer {

    /**
     * Create an explainer
     */
    public NaiveBayesExplainerImpl() {
    }
    
    /**
     * Make safe variable name (replacing 
     * @param varName 
     */
    public static final String safeStr(String varName) {
        return varName.replace("#", "_HASH_").replace("$", "_TAIL_").replace("+", "_PLUS_").replace("-", "_MINUS_")
                .replace("^", "_HEAD_").replace("*", "_STAR_").replace("/", "_SLASH_").replace(".", "_DOT_");
    }    

    @Override
    public IClassificationExplained explain(IClassification classification) throws ClassifyException {
        String[] formulae = new String[classification.getClassProbabilities().length];
        String[] algebraicExpressions = new String[classification.getClassProbabilities().length];
        String[][] featureNameAndValues = new String[classification.getClassProbabilities().length][];
        double[][] basicProbabilities = new double[classification.getClassProbabilities().length][];
        String pathGlobal = pathGlobal();
        String pathGlobalCountCategories = pathGlobalCountCategories();
        long globalCount = (classification.getExplanationData().containsKey(pathGlobal) ? classification.getExplanationData().get(pathGlobal) : 0);
        long globalCountCategories = (classification.getExplanationData().containsKey(pathGlobalCountCategories) ? classification.getExplanationData().get(pathGlobalCountCategories) : 0);
        double[] likelyhood = new double[classification.getClassProbabilities().length];
        double likelyhoodTot = 0; 
        for (int i = 0; i < classification.getClassProbabilities().length; i++) {
            featureNameAndValues[i] = new String[classification.getFeatures().size()];
            basicProbabilities[i] = new double[classification.getFeatures().size()];
            StringWriter formula = new StringWriter();
            StringWriter algebraicExpression = new StringWriter();
            String category = classification.getClassProbabilities()[i].getCategory();
            String pathCategory = pathCategory(category);
            long categoryCount = (classification.getExplanationData().containsKey(pathCategory) ? classification.getExplanationData().get(pathCategory) : 0);
            double product = 1.0d;
            int j=0;
            for (Map.Entry<String, String> feature : classification.getFeatures().entrySet()) {
                featureNameAndValues[i][j] = feature.getKey()+"="+feature.getValue();
                String pathFeatureKey = pathFeatureKey(feature.getKey());
                long featureCount = (classification.getExplanationData().containsKey(pathFeatureKey) ? classification.getExplanationData().get(pathFeatureKey) : 0);
                if (featureCount > 0) {
                    String pathCategoryFeatureKey = pathCategoryFeatureKey(category, feature.getKey());
                    long categoryFeatureCount = (classification.getExplanationData().containsKey(pathCategoryFeatureKey) ? classification.getExplanationData().get(pathCategoryFeatureKey) : 0);
                    String pathFeatureKeyCountValueTypes = pathFeatureKeyCountValueTypes(feature.getKey());
                    long featureCountValueTypes = (classification.getExplanationData().containsKey(pathFeatureKeyCountValueTypes) ? classification.getExplanationData().get(pathFeatureKeyCountValueTypes) : 0);
                    String pathCategoryFeatureKeyValue = pathCategoryFeatureKeyValue(category, feature.getKey(), feature.getValue());
                    long categoryFeatureValueCount = (classification.getExplanationData().containsKey(pathCategoryFeatureKeyValue) ? classification.getExplanationData().get(pathCategoryFeatureKeyValue) : 0);                    
                    if (classification.isLaplaceSmoothed()) {
                        double basicProbability = (categoryFeatureCount == 0 ? 0 : 1d * (categoryFeatureValueCount + classification.getLaplaceSmoothingAlpha()) / (categoryFeatureCount + featureCountValueTypes * classification.getLaplaceSmoothingAlpha()));
                        if (categoryFeatureCount == 0) {
                            formula.append(pathCategoryFeatureKey);
                            algebraicExpression.append("0");
                        } else {
                            formula.append("(" + safeStr( pathCategoryFeatureKeyValue ) + " + alpha)/(" + safeStr( pathCategoryFeatureKey ) + " + ( " + safeStr( pathFeatureKeyCountValueTypes ) + " * alpha ))");
                            algebraicExpression.append("(" + categoryFeatureValueCount + " + " + classification.getLaplaceSmoothingAlpha() + " )/(" + categoryFeatureCount + " + ( " + featureCountValueTypes + " * " + classification.getLaplaceSmoothingAlpha() + " ))");
                        }
                        product *= basicProbability;
                        basicProbabilities[i][j]=basicProbability;                        
                    } else {
                        double basicProbability = (categoryFeatureCount == 0 ? 0 : 1d * categoryFeatureValueCount / categoryFeatureCount);
                        if (categoryFeatureCount == 0) {
                            formula.append(safeStr(pathCategoryFeatureKey));
                            algebraicExpression.append("0");
                        } else {
                            formula.append(safeStr(pathCategoryFeatureKeyValue) + " / " + safeStr(pathCategoryFeatureKey));
                            algebraicExpression.append(categoryFeatureValueCount + " / " + categoryFeatureCount);
                        }
                        product *= basicProbability;
                        basicProbabilities[i][j]=basicProbability;
                    }
                    formula.append(" * ");
                    algebraicExpression.append(" * ");
                }
                j++;
            }
            formula.append("1 ");
            algebraicExpression.append("1 ");
            if (classification.isLaplaceSmoothed() && classification.isLaplaceSmoothedVariant()) {
                likelyhood[i] = ((1d * categoryCount + classification.getLaplaceSmoothingAlpha()) / (globalCount + globalCountCategories * classification.getLaplaceSmoothingAlpha())) * product;
                formulae[i] = "((" + safeStr(pathCategory) + " + alpha) / (" + safeStr(pathGlobal) + " + (" + safeStr(pathGlobalCountCategories) + " * alpha))) * (" + formula.toString() + ")";
                algebraicExpressions[i] = "((" + categoryCount + " + " + classification.getLaplaceSmoothingAlpha() + ") / (" + globalCount + " + (" + globalCountCategories + " * " + classification.getLaplaceSmoothingAlpha() + "))) * (" + algebraicExpression.toString() + ")";
            } else {
                likelyhood[i] = 1d * categoryCount / globalCount * product;
                formulae[i] = safeStr(pathCategory) + " / " + safeStr(pathGlobal) + " * (" + formula.toString() + ")";
                algebraicExpressions[i] = categoryCount + " / " + globalCount + " * (" + algebraicExpression.toString() + ")";
            }
            likelyhoodTot += likelyhood[i];
        }
        for (int i = 0; i < classification.getClassProbabilities().length; i++) {
            double proba = likelyhood[i] / likelyhoodTot;
            if (proba > 1d) {
                // could equal 1.000000000002 due to double precision issue;
                proba = 1d;
            } else if (proba < 0) {
                proba = 0d;
            }
            if (Math.abs(proba - classification.getClassProbabilities()[i].getProbability()) > EPSILON) {
                Logger.getLogger(getClass().getName()).warning("Class " + classification.getClassProbabilities()[i].getCategory() + " probability differs PExplained = " + proba + " <> P = " + classification.getClassProbabilities()[i].getProbability());
            }
        }
        return new ClassificationExplainedImpl(classification, likelyhood, formulae, algebraicExpressions, featureNameAndValues, basicProbabilities);
    }

    private static final double EPSILON = 0.00001;
}
