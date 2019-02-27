#ifndef KOALA_DEF_SCHEDULING_H
#define KOALA_DEF_SCHEDULING_H

/** \file schedule.h
 *  \brief Scheduling algorithms (optional).
 */

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "../base/defs.h"
#include "../container/assoctab.h"
#include "../algorithm/search.h"

namespace Koala
{
	/**\brief Auxiliary structures for scheduling.*/
    struct SchedulingStructs {

		/** \brief Single task.
		 *
		 *  The class of task keep some basic information: length of task, release time, due date and the pointer to the vertex associated with task.*/
		template< class GraphType > struct Task
		{
			int length/**\brief Length*/,release/**\brief Release time*/,duedate/**\brief Due date*/;
			typename GraphType::PVertex vertex;/**<\brief Pointer to vertex in graph of conflicts or digraph of precedence constraints.*/

			/** \brief Constructor 
			 *
			 * \param _vertex the value for vertex, i.e. the corresponding vertex in graph of conflicts.
			 * \param _length the length of new-created task.*/
			Task( typename GraphType::PVertex _vertex = 0,int _length =1): length( _length ), release( 0 ),
				duedate(std::numeric_limits< int >::max()), vertex(_vertex)
				{ }

			/** \brief Constructor
			 *
			 * \param _length the length of new-created task.
			 * \param _release the task release date.
			 * \param _duedate the due date of task.
			 * \param _vertex the value for vertex, i.e. the corresponding vertex in graph of conflicts.*/
			Task( int _length, int _release, int _duedate, typename GraphType::PVertex _vertex = 0 ):
				length( _length ), release( _release ), duedate( _duedate ), vertex( _vertex )
				{ }
		};

        class Schedule;
		/** \brief Scheduled task part.
		 *
		 *  The structure represent the continuous time interval associated with part of task.
		 *  This is an output structure. */
		struct TaskPart
		{
		    friend class Schedule;
			int task/**\brief Task index*/,start/**\brief Starting time*/,end/**\brief Finishing time*/,part/**\brief Part of task (preemptive tasks) index. */;
			/**\brief Constructor*/
			TaskPart( int _task = 0, int _start = 0, int _end = 0, int _part = 0):
				task( _task ), start( _start ), end( _end ), part( _part )
				{ }

            protected:
                struct compareEndTime
                {
                    bool operator() ( const TaskPart &first, const TaskPart &second )
                    {
                        return first.end < second.end;
                    }
                };

                struct compareIndexAndStartTime
                {
                    bool operator() ( const TaskPart &first, const TaskPart &second )
                    {
                        return first.task < second.task ||  (first.task == second.task && first.start < second.start);
                    }
                };
		};

		/** \brief Task window.
		 *
		 *  The output structure used by critical path algorithm. It represents  the time windows in which a task can be executed 
		 *  and the optimal schedule is feasible. */
		struct TaskWindow
		{
		    int earliestStart/**\brief Earliest task start.*/,earliestFinish/**\brief Earliest task completion. */,latestStart/**\brief Latest task start */,latestFinish/**\brief Latest task completion.*/;

			/**\brief Constructor.*/
			TaskWindow()
				{ }

			/**\brief Constructor.*/
			template< class GraphType > TaskWindow( Task< GraphType > task, int start, int finish ):
				earliestStart( start ), earliestFinish( start + task.length ),
				latestStart( finish - task.length ), latestFinish( finish )
				{ }
		};

		/** \brief Schedule.
		 *
		 *  The output structure keeps information about generated schedule.*/
		struct Schedule
		{
			typedef std::vector< TaskPart > Machine;/**<\brief Type of vector of TaskParts associated with machine*/
			typedef std::vector< Machine > Type;/**<\brief Type of vector of vectors (for each machine single vector of TaskParts).*/

			/** \brief Vector of vectors of TaskPart structures.
			 *    
             *  This an output structure that assigns to each machine (indexes from 0) a vector of TaskPart structures that represents parts of tasks executed on this machine.
			 *  Hence this is an STL vector of vectors of TaskParts. 
			 *  Idle blocks are not represented in this structure.*/
			Type machines; 

			/**\brief Constructor.
			 *
			 * The constructor that initializes the number of mahines.
			 * \param m the number of machines*/
			Schedule( int m = 0 )
				{ setMachNo( m ); }
			/**\brief Set number of machines.
			 *
			 * \param m the number of machines*/
			void setMachNo( int m )
				{ machines.clear(); machines.resize( m ); }
			/**\brief Get number of machines.*/
			int getMachNo()
				{ return machines.size(); }
			/**\brief Clear m-th machine.*/
			void clearMachine( int m )
				{ machines[m].clear(); }
			/**\brief Clear all machines.*/
			inline void clearMachines();

			/**\brief Get part index in vector.
			 *
			 *  The method returns the index of TaskPart executed in \a time slot on machine \a machNo*/
			inline int part( int machNo, int time );

			/**\brief Get Sequence of TaskParts sequences associated with tasks.*/
			template< typename IntInserter, typename STDPairOfIntInserter >
			void taskPartList( SearchStructs::CompStore<IntInserter,STDPairOfIntInserter> out );
		};


    };

	/** \brief Scheduling algorithms (parametrized).
	 *
	 *  The class provides some implementations of scheduling algorithms in the major part based on graphs.
	 *  Directed acyclic graphs (DAG) are  used to represent precedence constraints for task represented by vertices.
	 *  All the parameters associated with scheduling like due dates, lengths, weights are assumed to be integers.
	 *  \tparam DefaultStructs the class decides about the basic structures and algorithm. Can be used to parametrize algorithms.
	 *  \ingroup DMschedule   */
	template< class DefaultStructs > class SchedulingPar : public SchedulingStructs
	{
	public:

		/** \brief Sort by Longest Processing Time (LPT)
		 * 
		 *  \param[in] begin the iterator to first element of the container with tasks (Task).
		 *  \param[in] end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param[out] out the iterator to the output container with numbers that stand for the position of consecutive tasks from input in LPT sequence.
		 *  \return the number of tasks in input container and the length of sequence \a out. */
		template< typename TaskIterator, typename Iterator >
			static int sortLPT( TaskIterator begin, TaskIterator end, Iterator out )
			{
				return sortByComp< compareLPT< std::pair< TaskIterator,int > > >( begin,end,out );
			}

		/** \brief Sort by Shortest Processing Time SPT
		 *  
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param[out]  out the iterator to the output container with numbers that stand for the position of consecutive tasks from input in SPT sequence.
		 *  \return the number of tasks in the input container and the length of sequence \a out.*/
		template< typename TaskIterator, typename Iterator >
			static int sortSPT( TaskIterator begin, TaskIterator end, Iterator out )
			{
				return sortByComp< compareSPT< std::pair< TaskIterator,int > > >( begin,end,out );
			}

		/** \brief Sort by Earliest Due Date EDD.
		 *  
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param[out]  out the iterator to the output container with numbers that stand for the position of consecutive tasks from input in EDD sequence.
		 *  \return the number of tasks in the input container and the length of sequence out.*/
		template< typename TaskIterator, typename Iterator >
			static int sortEDD( TaskIterator begin, TaskIterator end, Iterator out )
			{
				return sortByComp< compareEDD< std::pair< TaskIterator,int > > >( begin,end,out );
			}

		/** \brief Calculate C<sub>max</sub> for \a schedule.
		 *
		 *  \param[in] begin the iterator to first element of the container with tasks (Task).
		 *  \param[in] end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param[out] schedule the reference to the achieved Schedule for tasks from the container. 
		 *  \return the makespan of the \a schedule.*/
		template< typename TaskIterator >
			static int CMax( TaskIterator begin, TaskIterator end, const Schedule &schedule );
		/** \brief Calculate ΣC<sub>j</sub> for \a schedule.
		 *
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param schedule the reference to the achieved Schedule for tasks from the container.
		 *  \return the sum of completion times for \a schedule.*/
		template< typename TaskIterator >
			static int SigmaCi( TaskIterator begin, TaskIterator end, const Schedule &schedule );
		/** \brief Calculate ΣT<sub>j</sub> for \a schedule.
		 *
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param schedule the reference to the achieved Schedule for tasks from the container.
		 *  \return the sum of tardinesses for \a schedule.*/
		template< typename TaskIterator >
			static int SigmaTi( TaskIterator begin, TaskIterator end, const Schedule &schedule );
		/** \brief Calculate ΣU<sub>j</sub> for \a schedule.
		 *
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param schedule the reference to the achieved Schedule for tasks from the container.
		 *  \return the number of latenesses for \a schedule.*/
		template< typename TaskIterator >
			static int SigmaUi( TaskIterator, TaskIterator, const Schedule &schedule );
		/** \brief Calculate L<sub>max</sub> for \a schedule.
		 *
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param schedule the reference to the achieved Schedule for tasks from the container.
		 *  \return the maximal lateness for \a schedule.*/
		template< typename TaskIterator >
			static int LMax( TaskIterator, TaskIterator, const Schedule &schedule );

		//Test poprawności danego harmonogramu
		/** \brief Test correctness of \a schedule.
		 *
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param DAG an directed acyclic graph representing the precedence constraints. Where each task is assigned vertex
		 *  and precedence is determined by arcs. 
		 *  \param schedule the reference to the tested Schedule object.
		 *  \param nonPmtn if true, tasks are non-preemptible. If false, tasks are preemptible.
		 *  \return true if \a schedule is correct, false if any kind of conflict arise. */
		template< typename GraphType, typename TaskIterator >
			static bool test( TaskIterator begin, TaskIterator end, const GraphType &DAG, const Schedule &schedule, bool nonPmtn = true );

		//-|prec|Cmax - czyli szeregowanie ścieżką krytyczną z zadaniami w wierzchołkach
		/** \brief Solve -|prec|C<sub>max</sub>.
		 *
		 *  For a given sequence of task and precenence constraints (directed acyclic graph) the method finds the schedule using critical path method.
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param DAG an directed acyclic graph representing the precedence constraints. Where each task is assigned vertex
		 *  and precedence is determined by arcs. 
		 *  \param[out] schedule the iterator to the container with SchedulingStructs::TaskWindow, that gives output schedule i.e. 
		 *  The container on i-th position stores i-th task possible time of execution.
		 *  \return the makespan.
		 *
		 *  [See example](examples/schedule/scheduling_critical.html). */
		template< typename GraphType, typename TaskIterator, typename TaskWindowIterator >
			static int critPath( TaskIterator begin, TaskIterator end, const GraphType &DAG, TaskWindowIterator schedule );

		/** \brief P|prec|- heuristic.
		 *  
		 *  The list scheduling algorithm which for task order given by input list inserts them into schedule.
		 *  Release dates are taken into account while due dates are ignored. 
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param DAG an directed acyclic graph representing the precedence constraints. Where each task is assigned a vertex
		 *  and precedence is determined by arcs. 
		 *  \param[out] schedule the reference to the Schedule object to which the schedule is written. 
		 *   Should be empty. Mind that this structure determine the number of processors.
		 *  \return the makespan of achieved schedule.	*/
		template< typename GraphType, typename TaskIterator >
			static int ls( TaskIterator begin, TaskIterator end, const GraphType &DAG, Schedule &schedule );

		/** \brief Solve P2|UET,prec|C<sub>max</sub>.
		 *
		 *  The problem is solved with Coffman-Graham algorithm.
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task). Task must be unit-length, release and due dates are ignored.
		 *  \param DAG an directed acyclic graph representing the precedence constraints. Where each task is assigned a vertex
		 *  and precedence is determined by arcs. 
		 *  \param[out] schedule the reference to the Schedule object to which the schedule is written. 
		 *  Should be empty on entrance and defined for two processors.
		 *  \return the makespan for P2|UET,prec|C<sub>max</sub>.
		 *
 *  [See example](examples/schedule/scheduling_coffman.html). */
		template< typename GraphType, typename TaskIterator >
			static int coffmanGraham( TaskIterator begin, TaskIterator end, const GraphType &DAG, Schedule &schedule );

		/** \brief Solve 1|pmtn,prec,r<sub>i</sub>|L<sub>max</sub>.
		 *
		 *  The problem is solved with Liu algorithm.
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param DAG an directed acyclic graph representing the precedence constraints. Where each task is assigned a vertex
		 *  and precedence is determined by arcs. 
		 *  \param[out] schedule the reference to the Schedule object to which the schedule is written. 
		 *  Should be empty on entrance and defined for single processor.
		 *  \return the minimal possible L<sub>max</sub> (maximal lateness).
		 *
		 *  [See example](examples/schedule/scheduling_precliu.html). */
		template< typename GraphType, typename TaskIterator >
			static int precLiu( TaskIterator begin, TaskIterator end, const GraphType &DAG, Schedule &schedule );

		/** \brief Solve P|UET,in-tree|L<sub>max</sub>.
		 *
		 *  The problem is solved with Brucker algorithm. It works for unit length tasks. Release dates are ignored. 
		 *  \param begin the iterator to first element of the container with tasks (Task). 
		 *  \param end the iterator to past-the-end element of the container with tasks (Task).
		 *  \param DAG an directed acyclic graph representing the precedence constraints. Where each task is assigned a vertex
		 *  and precedence is determined by arcs. The graph must be an in-forest.
		 *  \param[out] schedule the reference to the Schedule object to which the schedule is written. 
		 *   Should be empty. Mind that this structure determine the number of processors.
		 *  \return the minimal possible L<sub>max</sub> (maximal lateness).
		 *
		 *  [See example](examples/schedule/scheduling_brucker.html). */
		template< typename GraphType, typename TaskIterator >
			static int brucker( TaskIterator begin, TaskIterator end, const GraphType &DAG, Schedule &schedule );

		/** \brief Solve P|UET,in-tree|C<sub>max</sub>.
		 *
		 *  The problem is solved with Hu algorithm. 
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task). 
		 *  \param DAG an directed acyclic graph representing the precedence constraints  Where each task is assigned a vertex
		 *  and precedence is determined by arcs. The graph must be an in-forest.
		 *  \param[out] schedule the reference to the Schedule object to which the schedule is written. 
		 *   Should be empty on entrance. Mind that this structure determine the number of processors.
		 *  \return the makespan of optimal schedule.
		 *
		 *  [See example](examples/schedule/scheduling_hu.html). */
		template< typename GraphType, typename TaskIterator >
			static int hu( TaskIterator begin, TaskIterator end, const GraphType &DAG, Schedule &schedule );

		/** \brief Solve P||ΣC<sub>i</sub>.
		 *
		 *  Release and due dates are ignored.
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task). 
		 *  \param[out] schedule the reference to the Schedule object to which the schedule is written.
		 *   Should be empty on entrance. Mind that this structure determine the number of processors.
		 *  \return the minimal sum of completion times for P||ΣC<sub>i</sub>*/
		template< typename TaskIterator > static int spt( TaskIterator begin, TaskIterator end, Schedule &schedule );

		/** \brief Solve 1||ΣU<sub>i</sub>.
		 *
		 *  The method solves the problem using the Hodgson algorithm. The release dates are ignored.
		 *  \param begin the iterator to first element of the container with tasks (Task).
		 *  \param end the iterator to past-the-end element of the container with tasks (Task). 
		 *  \param[out] schedule the reference to the Schedule object to which the optimal schedule is written. 
		 *  Should be empty on entrance. And defined for single processor.
		 *  \return the minimal number of late tasks.
		 *
		 *  [See example](examples/schedule/scheduling_hodgson.html). */
		template< typename TaskIterator > static int hodgson( TaskIterator begin, TaskIterator end, Schedule &schedule );

	protected:
		template< typename Comp, typename TaskIterator, typename Iterator >
			static int sortByComp( TaskIterator begin, TaskIterator end, Iterator out );

		template< typename Pair > struct compareLPT
		{
			bool operator()( Pair a, Pair b )
			{
				return a.first->length > b.first->length || (a.first->length == b.first->length && a.second < b.second);
			}
		};

		template< typename Pair > struct compareSPT
		{
			bool operator()( Pair a, Pair b )
			{
				return a.first->length < b.first->length || (a.first->length == b.first->length && a.second < b.second);
			}
		};

		template< typename Pair > struct compareEDD
		{
			bool operator()( Pair a, Pair b )
			{
				return a.first->duedate < b.first->duedate || (a.first->duedate == b.first->duedate && a.second < b.second);
			}
		};

		template< typename Pair > struct compareSecondFirst
		{
			bool operator()( Pair a, Pair b )
			{
				return a.second < b.second;
			}
		};

		template< typename Pair > struct compareSecondLast
		{
			bool operator()( Pair a, Pair b )
			{
				return a.second > b.second;
			}
		};

		template< typename Task > struct Triple
		{
			Task task;
			int start, finish;

			Triple( Task _task = Task()): task( _task )
				{ }
		};

		template< typename Task > struct Element
		{
			Task task;
			int index,priority,duedate,degree,timeleft,parts;

			Element( Task _task = Task(), int _index = 0 ): task( _task ), index( _index ), priority( 0 ),
				duedate( 0 ), timeleft( _task.length ), parts( 0 )
				{ }
		};

		struct HodgsonElement
		{
			int index,length,duedate;
			bool late;

			HodgsonElement( int _index = 0, int _length = 1, int _duedate = 0 ): index( _index ), length( _length ),
				duedate( _duedate ), late( 0 )
				{ }
		};
	};
	/** \brief Scheduling algorithms (default).
	 *
	 *  The class provides some implementations of scheduling algorithms in the major part based on graphs.
	 *  All the parameters associated with scheduling like due dates, length, weights are assumed to be integers.
	 *  The simpler default  version of SchedulingPar in which DefaultStructs = AlgsDefaultSettings.
	 *  \ingroup DMschedule   */
	class Scheduling: public SchedulingPar< AlgsDefaultSettings > { };

#include "schedule.hpp"
}

#endif
