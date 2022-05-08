package com.namsor.oss.classify.bayes;

import java.io.StringWriter;
import java.util.*;

/**
 * The detailed explanation of a classification :
 * - likelyhood values
 * - likelyhood formulae (in a readable format)
 * - likelyhood expressions (in a readable format)
 * The toString() function generates a JavaScript that can interpreted
 *
 * @author elian
 */
public class ClassificationExplainedImpl implements IClassificationExplained {
    
    private static final Comparator KEY_ORDER = (Comparator) (Object o1, Object o2) -> {
        Map.Entry<String, Long> e1 = (Map.Entry<String, Long>) o1;
        Map.Entry<String, Long> e2 = (Map.Entry<String, Long>) o2;
        return e1.getKey().compareTo(e2.getKey());
    };

    private final IClassification classification;
    private final double[] likelyhoods;
    private final String[] likelyhoodFormulae;
    private final String[] likelyhoodExpressions;

    /**
     * Create an immutable detailed explanation of a classification :
     *
     * @param classification        The classification output and explainData
     * @param likelyhoods           The likelyhood values
     * @param likelyhoodFormulae    The likelyhood formulae
     * @param likelyhoodExpressions The likelyhood expressions
     */
    public ClassificationExplainedImpl(IClassification classification, double[] likelyhoods, String[] likelyhoodFormulae, String[] likelyhoodExpressions) {
        this.classification = classification;
        this.likelyhoods = likelyhoods;
        this.likelyhoodFormulae = likelyhoodFormulae;
        this.likelyhoodExpressions = likelyhoodExpressions;
    }

    /**
     * @return the classification
     */
    public IClassification getClassification() {
        return classification;
    }

    /**
     * @return the likelyhoods
     */
    @Override
    public double[] getLikelyhoods() {
        return likelyhoods;
    }

    /**
     * @return the likelyhoodFormulae
     */
    @Override
    public String[] getLikelyhoodFormulae() {
        return likelyhoodFormulae;
    }

    /**
     * @return the likelyhoodExpressions
     */
    @Override
    public String[] getLikelyhoodExpressions() {
        return likelyhoodExpressions;
    }

    @Override
    public IClassProbability[] getClassProbabilities() {
        return getClassification().getClassProbabilities();
    }

    @Override
    public Map<String, Long> getExplanationData() {
        return getClassification().getExplanationData();
    }

    @Override
    public boolean isLaplaceSmoothed() {
        return getClassification().isLaplaceSmoothed();
    }

    @Override
    public boolean isLaplaceSmoothedVariant() {
        return getClassification().isLaplaceSmoothedVariant();
    }

    @Override
    public Map<String, String> getFeatures() {
        return getClassification().getFeatures();
    }

    @Override
    public double getLaplaceSmoothingAlpha() {
        return getClassification().getLaplaceSmoothingAlpha();
    }

    @Override
    public String toJavaScriptText() {
        StringWriter sw = new StringWriter();
        sw.append("// JavaScript : " + "\n");
        if (isLaplaceSmoothed()) {
            sw.append("// laplaced smoothing alpha " + "\n");
            sw.append("var alpha=" + getLaplaceSmoothingAlpha() + "\n");
            sw.append("\n// laplaced smoothing variant : " + isLaplaceSmoothedVariant() + "\n");
        }
        sw.append("\n// observation table variables " + "\n");
        List<Map.Entry<String, Long>> entries = new ArrayList(getExplanationData().entrySet());
        Collections.sort(entries, KEY_ORDER);
        for (Map.Entry<String, Long> entry : entries) {
            sw.append("var " + entry.getKey() + "=" + entry.getValue() + "\n");
        }
        sw.append("\n\n// likelyhoods by category " + "\n");
        StringWriter swLikelyhoodTot = new StringWriter();
        for (int i = 0; i < getClassProbabilities().length; i++) {
            sw.append("\n// likelyhoods for category " + getClassProbabilities()[i].getCategory() + "\n");
            sw.append("var likelyhoodOf" + getClassProbabilities()[i].getCategory() + "=" + getLikelyhoodFormulae()[i] + "\n");
            sw.append("var likelyhoodOf" + getClassProbabilities()[i].getCategory() + "Expr=" + getLikelyhoodExpressions()[i] + "\n");
            sw.append("var likelyhoodOf" + getClassProbabilities()[i].getCategory() + "Value=" + getLikelyhoods()[i] + "\n");
            swLikelyhoodTot.append("likelyhoodOf" + getClassProbabilities()[i].getCategory() + "+");
        }
        sw.append("\n\n// probability estimates by category " + "\n");
        StringWriter swProbabilityTot = new StringWriter();
        for (int i = 0; i < getClassProbabilities().length; i++) {
            sw.append("\n// probability estimate for category " + getClassProbabilities()[i].getCategory() + "\n");
            sw.append("var probabilityOf" + getClassProbabilities()[i].getCategory() + "=" + "likelyhoodOf" + getClassProbabilities()[i].getCategory() + "/(" + swLikelyhoodTot.toString() + "0)" + "\n");
            sw.append("var probabilityOf" + getClassProbabilities()[i].getCategory() + "Value=" + getClassProbabilities()[i].getProbability() + "\n");
            swProbabilityTot.append("probabilityOf" + getClassProbabilities()[i].getCategory() + " + ");
        }

        sw.append("\n\n// return the highest probability estimate for evaluation " + "\n");
        sw.append("probabilityOf" + getClassProbabilities()[0].getCategory());
        return sw.toString();        
    }
    

    @Override
    public String toString() { 
        return toJavaScriptText();
    }

    @Override
    public boolean isUnderflow() {
        return getClassification().isUnderflow();
    }

}
