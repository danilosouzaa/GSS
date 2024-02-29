/**
 * @file	gaproblem.h
 *
 * @brief	Handle a general optimization problem
 *
 * @author	Danilo and Igor
 * @date    
 */


#include <limits.h>
#include "gpulib/types.h"

#include "gapparams.h"
#include "consts.h"


#ifndef __gaproblem_h
#define __gaproblem_h

// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //


// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //

/*
 * Classes
 */
class GAPSolution;


struct GAPJobsData{
    ushort id;
    ushort idAgentAllocated;
    uint *costAgentsAllocation; //cost of allocating the task to any agent
    uint *resourceConsumed; //resource to be consumed to allocate the job to any agent
};
typedef GAPJobsData  *PGAPJobsData;// pointer list jobs

struct GAPAgentsData{
    ushort id;
    uint capacity; //maximum resource capacity of the agent
    uint resUsage; // amount of resource already allocated
    bool *jobsAllocated; // true - if the task is allocated to the agent, false -otherwise
};
typedef GAPAgentsData  *PGAPAgentsData;

// ################################################################################ //
// ##                                                                            ## //
// ##                               CLASS GAProblem                              ## //
// ##                                                                            ## //
// ################################################################################ //

/*!
 * Class GAProblem
 */


class GAProblem
{
public:
    //GAPParams       &params;                 ///< GAP parameters
    bool          	costTour;             ///< Cost calculation method (tour/path)
	bool          	distRound;            ///< Sum 0.5 to euclidean distance calculation?
	bool          	coordShift;           ///< Shift clients coordinates if necessary

	
    char            filename[OFM_LEN_PATH]; ///< Instance filename
    char            name[OFM_LEN_NAME];     ///< Instance filename

    ullong          timeLoad;               ///< Instance load time
    uint            maxWeight;              ///< Max weight value
    uint            maxCapacity;
    GAPJobsData    *jobs;                   ///< Jobs data
    uint            nJobs;                  ///< Jobs size
    GAPAgentsData   *agents;                 ///< Agents data
    uint            mAgents;                 ///< Agents size

    int             shiftX;                 ///< X coordinate shift, if any
    int             shiftY;                 ///< Y coordinate shift, if any

protected:
    /*!
     * Allocate memory for jobs and agents data.
     *
     * @param   nJobs    number of jobs
     * @param   mAgents  number of agents
     */
    void
    allocData(uint nJobs,uint mAgents);

public:
    /*!
     * Create an empty GAProblem instance.
     */
    GAProblem(bool _costTour, bool _distRound, bool _coordShift):
    	costTour(_costTour), distRound(_distRound), coordShift(_coordShift)  {
    	//(MLParams &pars) : params(pars) {

        jobs = NULL;
        agents = NULL;
        nJobs = 0;
        mAgents = 0;
        timeLoad = 0;
        maxWeight = 0;
        maxCapacity = 0;
    }
    /*!
     * Destroy GAProblem instance.
     */
    virtual ~GAProblem() {
        free();
    }
    /*!
     * Was coordinates shifted?
     */
    bool coordShifted() {
        return shiftX || shiftY;
    }
    /*!
     * Release problem instance.
     */
    void free();
    /*!
     * Load from file a GAP problem instance.
     *
     * @param   fname   Instance filename
     */
    void
    load(const char *fname);
    /*!
     * Save PFCL to a file. If \a fname is NULL, then the
     * loaded instance filename is used replacing the extension
     * by 'dat'
     *
     * @param   fname   Instance filename
     */

     void showInstance();
    void
    save(const char *fname = NULL);
    /*!
     * Create a new instance of a problem solution.
     *
     * @return  Returns a pointer to a new problem solution.
     */
    // GAProblem *
    // createSolution();
    // /*!
    //  * Friend classes
    //  */
    // friend class MLSolution;
};










#endif	// __mlproblem_h
