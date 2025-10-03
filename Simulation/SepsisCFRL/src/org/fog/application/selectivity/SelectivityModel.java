package org.fog.application.selectivity;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;

/**
 * Class representing the input-output relationships of application modules.
 * @author Harshit Gupta
 *
 */
public interface SelectivityModel {

	/**
	 * Function called to check whether incoming tuple can generate an output tuple.
	 * @return true if a tuple can be emitted (selection possible)
	 */
	public boolean canSelect();
	
	/**
	 * Average number of tuples generated per incoming input tuple.
	 * @return avg tuple generation rate
	 */
	public double getMeanRate();
	
	/**
	 * Maximum number of tuples generated per incoming input tuple.
	 * @return max tuple generation rate
	 */
	public double getMaxRate();

	/**
	 * Set the current tuple and device for selectivity decision
	 * @param tuple The current tuple being processed
	 * @param device The fog device processing the tuple
	 */
	default void setCurrentTupleAndDevice(Tuple tuple, FogDevice device) {
		// Default empty implementation
	}
}
