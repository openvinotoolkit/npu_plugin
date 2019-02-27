#ifndef KOALA_SAT2CNF_H
#define KOALA_SAT2CNF_H

/** \file sat2cnf.h
 *  \brief 2-satisfiability (optional).*/

#include "../graph/graph.h"
#include "../container/hashcont.h"
#include "../algorithm/search.h"
#include "../base/defs.h"

namespace Koala
{
	/** \brief 2-SAT solution.
	 *
	 *  Class contains methods for solving 2-SAT problem according to the algorithm described in:\n
	 *  "A linear-time algorithm for testing the truth of certain quantified boolean formulas", B. Aspvall, M.F. Plass, R.E. Tarjan,
	 *  Information Processing Letters, Volume 8, Issue 3, Pages 121–123, 1979. */
	template<class DefaultStructs> class Sat2CNFPar
	{
	public:

		/** \brief Literal type.
		 *
		 * Type Literal represents an literal that can appear in a clauses in 2-SAT problem:
		 * - int  - nonnegative variable number,
		 * - bool  - true means that the variable is in plain form (without negation),
		 * - false means that the variable is appears with negation. */
		typedef std::pair<int, bool> Literal;
		
		/** \brief Clause type.
		 *
		 * Type Class represents a clause (i.e., alternative of two literals) that can appear in a clauses in 2-SAT problem */
		typedef std::pair<Literal, Literal> Clause;

		/** \brief Get variables.
		 *
		 * The method returns the number of distinct variables.
		 * Iter should be an iterator over a collection of Clause objects.
		 * IterOut should be an iterator over a collection of int variables.
		 * \param begin - points to the beginning of a collection of clauses,
		 * \param end - points to the past-the-end element for a collection of clauses.
		 * \param out - output iterator for storing numbers of variables in increasing order.*/
		template<class Iter, class IterOut>
		static int vars(Iter begin, Iter end, IterOut out);

		/** \brief Solve 2-SAT.
		 *
		 * The method returns true if 2-SAT problem given as input has a solution, otherwise it returns false.
		 * Iter should be an iterator over a collection of Clause objects.
		 * IterOut should be an iterator over a collection of bool variables.
		 * \param begin - points to the beginning of a collection of clauses,
		 * \param end - points to the past-the-end element for a collection of clauses.
		 * \param out - output iterator for storing bool variables (values of variables in increasing order)
		 *  that represent the solution if one exists.
		 */
		template<class Iter, class IterOut>
		static bool solve(Iter begin, Iter end, IterOut out);

		/** \brief Evaluate.
		 * The method returns the value of a 2-SAT problem given as input for the realization of variables given as input by begin2 and end2 pointers.
		 * Iter should be an iterator over a collection of Clause objects.
		 * Iter2 should be an iterator over a collection of bool variables.
		 * \param begin - points to the beginning of a collection of clauses,
		 * \param end - points to the past-the-end element for a collection of clauses.
		 * \param begin2 - points to the beginning of a collection of bool values,
		 * \param end2 - points to the past-the-end element for a collection of bool values.*/
		template<class Iter, class Iter2>
		static bool eval(Iter begin, Iter end, Iter2 begin2, Iter2 end2);

	};

	class Sat2CNF : public Sat2CNFPar<AlgsDefaultSettings> {};


#include "sat2cnf.hpp"
}

#endif
