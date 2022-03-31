package intrusion_detection;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import algorithms.ContiguousCountingDAG;
import algorithms.PatternTrie;
import algorithms.RChunkPatterns;
import alphabets.Alphabet;
import alphabets.AminoAcidAlphabet;
import alphabets.BinaryAlphabet;
import alphabets.BinaryLetterAlphabet;
import alphabets.DegenerateAminoAcidAlphabet;
import alphabets.LatinAlphabet;
import be.cylab.java.roc.CurveCoordinates;
import be.cylab.java.roc.Main;
import be.cylab.java.roc.Roc;
import be.cylab.java.roc.Utils;
import util.Debug;
import util.Settings;


public class IntrusionDetection {
	// Parameters to use
	static int n = 7;	// Length of strings in self set
	static int r = 3; 	// Parameter r <= n
	static int r2 = 0; // Output k-th component of matching profile (0 for full profile)
	static String self = "syscalls\\snd-cert\\snd-cert.train";		// File containing self set (1 string per line)
	static String alpha = "syscalls\\\\snd-cert\\\\snd-cert.alpha"; // Alphabet, currently one of [infer|binary|binaryletter|amino|damino|latin]. Default: infer (uses all characters from \\\"self\\\" file as alphabet). Alternatively, specify file://[f] to set the alphabet to all characters found in file [f].
	static boolean invertmatch = false;	
	static boolean matching_profile = false;// true if r2 > 0
	static boolean logarithmize = true;		// Output logarithms instead of actual values
	static boolean counting = true;			// Count matching detectors instead of binary match
	
	// Class uses chunk matching only
	
	public static double getMatchingDetectors(String line, int i1, int i2, PatternTrie matcher, ContiguousCountingDAG counter, long baseline, List<PatternTrie> chunks) {
		int lineindex;
		if( line.length() < n ){
			System.out.print( "NaN" );
		}
		double score = 0;
		for( lineindex = 0; lineindex <= line.length()-n ; lineindex ++ ){
			double[] nmatch = new double[i2-i1+1];
			String l = line.substring( lineindex, lineindex+n );
			
			// r-chunk detectors
			int i = 0;
			for( PatternTrie chunkmatcher : chunks ){
				if( chunkmatcher.matches(l.substring(i),r) >= r != invertmatch ){
					nmatch[0] ++;
				}
				i++;
			}
			if( logarithmize ){
				nmatch[0] = Math.log(1+nmatch[0])/Math.log(2.);
			}
			
			score += nmatch[0];
		}
		
		System.out.println(score / lineindex);
		
        // Return averaged nmatch
	    return score / lineindex;
	}
	
	public static void negativeSelection(String file_path) throws FileNotFoundException {
		Settings.DEBUG = false;	// Print debug information
		
		// Set the alphabet to all characters found in file 
		Alphabet.set(new BinaryAlphabet());
		if (!new File(self).canRead() || !new File(alpha).canRead()) {
			throw new IllegalArgumentException("Can't read file " + self + " or " + alpha);
		}
		Alphabet.set(new Alphabet(new File(alpha)));
		
		if (r < 0 || n <= 0 || r > n) {
			throw new IllegalArgumentException(
					"Illegal value(s) for n and/or r");
		}
	
		Debug.log("constructing matcher");
		List<PatternTrie> chunks = RChunkPatterns.rChunkPatterns(self, n, r, 0);
		PatternTrie matcher = null;
		ContiguousCountingDAG counter = null;
		long baseline = 0; 
	
		// output matching lengths to be used
		int i1 = 0, i2 = 0;
		if( !matching_profile ){
			i1 = r; i2 = r; 
		} else {
			if( r2 > 0 && r2 <= n ){
				i1 = r2; i2 = r2;
			} else {
				i1 = 0; i2 = n;
			}
		}

		Debug.log("matcher constructed");
		
		// Scan english.test and generates scores
		File test_file = new File(file_path + ".test");
		File label_file = new File(file_path + ".labels");
		Scanner scan_test = new Scanner(test_file);
		Scanner scan_labels = new Scanner(label_file);
		List<Double> score = new ArrayList<Double>();
		List<Double> true_alert = new ArrayList<Double>();
		int i = 0;
		while ( scan_test.hasNextLine() ) {
			i++;
			String line = scan_test.nextLine().trim();
			score.add(getMatchingDetectors(line, i1, i2, matcher, counter, baseline, chunks));
			
			// for true_alert, use .labels
			String label = scan_labels.nextLine().trim();
			System.out.println(label);
			true_alert.add(Double.parseDouble(label));
		}
		
		// Close scanners
		scan_test.close();
		scan_labels.close();
		
		// normalize scores and map to double array
		double max_score = Collections.max(score);
		double[] score_array = score.stream().mapToDouble(d -> d / max_score).toArray();
		double[] true_alert_array = true_alert.stream().mapToDouble(d -> d).toArray();
		
		// Generate ROC and compute AUC
		Roc roc = new Roc(score_array, true_alert_array);
		System.out.println(roc.computeAUC());

		roc.computeRocPointsAndGenerateCurve("ROC-curve.png");
	}
	
	
	public static void main(String[] args) {
		try {
			String file_path = "syscalls\\snd-cert\\snd-cert.3";
			negativeSelection(file_path);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
