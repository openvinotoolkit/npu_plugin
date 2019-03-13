#ifndef DEF_MIS_H
#define DEF_MIS_H

/** \file mis.h
 *  \brief Maximum independent set and related problems (optional). */

#include "../base/defs.h"
#include "../base/rnd.h"
#include "../graph/graph.h"
#include "search.h"

namespace Koala
{

	/** \brief Vertex choosing strategies for maximal stable set, vertex cover and maximum clique heuristics. 
	 *
	 *  The namespace contains functors
	 *  that chooses a vertex basing on some specific rules.
	 *  Such rules could be: 
	 *  - first vertex, 
	 *  - random vertex, 
	 *  - vertex that meets specific requirements (has the largest degree for example).
	 *
	 *  These function objects may be used for example within \a getWMin and \a getWMax methods in Koala::MaxStableHeurPar 
	 *  to choose one vertex in each algorithm step. 
	 *  
	 *  They should be used with simple undirected graphs.
	 *
	 *  Each object function overload two parameter call function operator that takes
	 *  - \a g the copy of considered graph 
	 *  - associative container \a vertTab, which assigns integer weight to each vertex. However some strategies (First, Rand, GMin, GMax) ignore this parameter.
	 *
	 *  The functor returns one chosen vertex. 
	 *  \ingroup DMmis */
	namespace MaxStableStrategy
	{
	    namespace Privates {
            struct WMin_Strategy_tag {};
            struct WMax_Strategy_tag {};
            struct Strategy_tag : public WMin_Strategy_tag, WMax_Strategy_tag{};
	    }

		/* ----------------------------------------------------------------------
		*
		* Template:        N/A
		* Choice function: GetFirst
		* Function:        v[0]
		*
		*/
		/** \brief Get first vertex functor.
		 *
		 *  The for a graph functor returns always the first vertex on vertex list. Functor can be used in both approaches.*/
		class First : public Privates::Strategy_tag
		{
		public:
			/* \brief Call function operator.
			 *
			 *  \param g the copy of considered graph with pointer of original vertices in info field.
			 *  \param vertTab the associative array that assigns weights to vertices.(ignored)*/
			template< class GraphType, class VertContainer > typename GraphType::PVertex
				operator()( const GraphType &g, const VertContainer& vertTab )
				{
					(void)(vertTab); return g.getVert();
				}
		};

		/* ----------------------------------------------------------------------
		*
		* Template:        N/A
		* Choice function: RandVertex
		* Function:        v[?]
		*
		*/
		/** \brief Get random vertex functor.
		 *
		 *  The for a graph functor returns random vertex.  Functor can be used in both approaches.
		 *  \tparam RndGen numbers generator class.*/
		template <class RndGen=Koala::StdRandGen<> >
		class Rand : public Privates::Strategy_tag
		{
		public:

			/** \brief Constructor
			 *
			 * Constructor that initializes random numbers generator. 
			 * Use one of generators from header random or Koala::StdRandGen for lower c++ standards.*/
		    Rand(RndGen& rg) : rgen(&rg) {}

			/* \brief Call function operator.
			 *
			 *  \param g the copy of considered graph with pointer of original vertices in info field.
			 *  \param vertTab the associative array that assigns weights to vertices (ignored).*/
			template< class GraphType, class VertContainer > typename GraphType::PVertex
				operator()( const GraphType &g, const VertContainer& vertTab );

            private: RndGen* rgen;
		};

		/* ----------------------------------------------------------------------
		 *
		 * Template:        WMIN
		 * Choice function: GMIN
		 * Function:        1/( deg(v) + 1 )
		 *
		 * Notes:           Does not require vertices weights.
		 *
		 */
		/** \brief Get minimum degree vertex functor.
		 *
		 *  The for a graph (weights are ignored) functor returns vertex with minimal degree. It is advised to use with method \a getWMin.
		 *  \tparam RndGen numbers generator class.*/
		template <class RndGen=Koala::StdRandGen<> >
		class GMin : public Privates::WMin_Strategy_tag
		{
		public:
			/** \brief Constructor
			*
			* Constructor that initializes random numbers generator.
			* Use one of generators from header random or Koala::StdRandGen for lower c++ standards.*/
			GMin(RndGen& rg) : rgen(&rg) {}

			/* \brief Call function operator.
			 *
			 *  \param g the copy of considered graph with pointer of original vertices in info field.
			 *  \param vertTab the associative array that assigns weights to vertices.(ignored)*/
			template< class GraphType, class VertContainer > typename GraphType::PVertex
				operator()( const GraphType &g, const VertContainer& vertTab );

            private: RndGen* rgen;
		};

		/* ----------------------------------------------------------------------
		 *
		 * Template:        WMIN
		 * Choice function: GWMIN
		 * Function:        W(v)/( deg(v) + 1 )
		 *
		 */
		/** \brief Get minimum degree and maximum weight vertex functor.
		 *
		 *  The for a graph and weights functor returns vertex for which W(v)/( deg(v) +1) is maximal. It is advised to use with method \a getWMin.
		 *  \tparam RndGen numbers generator class.*/
		template <class RndGen=Koala::StdRandGen<> >
		class GWMin: public Privates::WMin_Strategy_tag
		{
		public:
			/** \brief Constructor
			*
			* Constructor that initializes random numbers generator.
			* Use one of generators from header random or Koala::StdRandGen for lower c++ standards.*/
		    GWMin(RndGen& rg) : rgen(&rg) {}

			/* \brief Call function operator.
			 *
			 *  \param g the copy of considered graph with pointer of original vertices in info field.
			 *  \param vertTab the associative array that assigns weights to vertices.*/
			template< class GraphType, class VertContainer > typename GraphType::PVertex
				operator()( const GraphType &g, const VertContainer& vertTab );

            private: RndGen* rgen;
		};

		/* ----------------------------------------------------------------------
		 *
		 * Template:        WMIN
		 * Choice function: GGWMIN
		 * Function:        All vertices in I satisfy:
		 *                  SUM of all u in N[v]  W(u)/(deg(u) + 1) <= W(v)
		 *
		 */
		/** \brief Get minimum degree vertex with weight functor.
		 *
		 *  The for a graph and weights functor returns vertices \a v for which Σ<sub>u ∈ N[v]</sub>W(u)/( deg(u) +1) ≤ W(v) . It is advised to use with method \a getWMin.
		 *  \tparam RndGen numbers generator class.*/
		template <class RndGen=Koala::StdRandGen<> >
		class GGWMin : public Privates::WMin_Strategy_tag
		{
		public:
			/** \brief Constructor
			 *
			 * Constructor that initializes random numbers generator.
			 * Use one of generators from header random or Koala::StdRandGen for lower c++ standards.*/
		    GGWMin(RndGen& rg) : rgen(&rg) {}

			/* \brief Call function operator.
			 *
			 *  \param g the copy of considered graph with pointer of original vertices in info field.
			 *  \param vertTab the associative array that assigns weights to vertices.*/
			template< class GraphType, class VertContainer > typename GraphType::PVertex
				operator()( const GraphType &g, const VertContainer& vertTab );

            private: RndGen* rgen;
		};

		/* ----------------------------------------------------------------------
		*
		* Template:        WMIN
		* Choice function: GWMIN2
		* Function:        W(v) / ( SUM of all u in N[v]  W(u) )
		*
		*/
		/** \brief Get maximum weight and minimum sum of neighbors weights vertex functor.
		 *
		 *  The for a graph and weights functor returns vertex for which W(v)/Σ<sub>u ∈ N[v]</sub>W(u) is maximal. It is advised to use with method \a getWMin.
		 *  \tparam RndGen numbers generator class.*/
		template <class RndGen=Koala::StdRandGen<> >
		class GWMin2 : public Privates::WMin_Strategy_tag
		{
		public:
		    /** \brief Constructor
			 *
			 * Constructor that initializes random numbers generator. 
			 * Use one of generators from header random or Koala::StdRandGen for lower c++ standards.*/
		    GWMin2(RndGen& rg) : rgen(&rg) {}

			/* \brief Call function operator.
			 *
			 *  \param g the copy of considered graph with pointer of original vertices in info field.
			 *  \param vertTab the associative array that assigns weights to vertices.*/
			template< class GraphType, class VertContainer > typename GraphType::PVertex
				operator()( const GraphType &g, const VertContainer& vertTab );

            private: RndGen* rgen;
		};

		/* ----------------------------------------------------------------------
		*
		* Template:        WMAX
		* Choice function: GMAX
		* Function:        1/( deg(v) * (deg(v) + 1) )
		*
		* Notes:           Does not require vertices weights.
		*
		*/
		/** \brief Get maximum degree vertex functor.
		 *
		 *  The for a graph gets the vertex with maximum degree. It is advised to use this functor with function \a getWMax.
		 *  \tparam RndGen numbers generator class.*/
		template <class RndGen=Koala::StdRandGen<> >
		class GMax : public Privates::WMax_Strategy_tag
		{
		public:
		    /** \brief Constructor
			 *
			 * Constructor that initializes random numbers generator. 
			 * Use one of generators from header random or Koala::StdRandGen for lower c++ standards.*/
		    GMax(RndGen& rg) : rgen(&rg) {}

			/* \brief Call function operator.
			 *
			 *  \param g the copy of considered graph with pointer of original vertices in info field. 
			 *  \param vertTab the associative array that assigns weights to vertices (ignored).*/
			template< class GraphType, class VertContainer > typename GraphType::PVertex
				operator()( const GraphType &g, const VertContainer& vertTab );

            private: RndGen* rgen;
		};

		/* ----------------------------------------------------------------------
		*
		* Template:        WMAX
		* Choice function: GWMAX
		* Function:        W(v)/( deg(v) * (deg(v) + 1) )
		*
		*/
		/** \brief Get maximum degree and minimal weight vertex functor.
		 *
		 *  The for a graph gets the vertex for which the function W(v) / (deg(v)*(deg(v)-1)) is minimal. It is advised to use this functor with function \a getWMax.
		 *  \tparam RndGen numbers generator class.*/
		template <class RndGen=Koala::StdRandGen<> >
		class GWMax: public Privates::WMax_Strategy_tag
		{
		public:
			/** \brief Constructor
			*
			* Constructor that initializes random numbers generator.
			* Use one of generators from header random or Koala::StdRandGen for lower c++ standards.*/
			GWMax(RndGen& rg) : rgen(&rg) {}

			/* \brief Call function operator.
			 *
			 *  \param g the copy of considered graph with pointer of original vertices in info field.
			 *  \param vertTab the associative array that assigns weights to vertices.*/
			template< class GraphType, class VertContainer > typename GraphType::PVertex
				operator()( const GraphType &g, const VertContainer& vertTab );

            private: RndGen* rgen;
		};

		/* ----------------------------------------------------------------------
		*
		* Template:        WMAX
		* Choice function: GGWMAX
		* Function:        All vertices in I satisfy:
		*                  SUM of all u in N[v]  W(u) /( deg(u) * (deg(u) + 1) )  >=  W(v) /( deg(v) + 1 )
		*
		*/
		/** \brief Get maximum degree and minimal weight vertex functor.
		 *
		 *  The for a graph and weights gets the vertices v that satisfy  Σ<sub>u ∈ N[v]</sub> W(u) / (deg(u)*(deg(u)-1)) ≥ W(v) /( deg(v) + 1 ). It is advised to use this functor with function \a getWMax.
		 *  \tparam RndGen numbers generator class.*/
		template <class RndGen=Koala::StdRandGen<> >
		class GGWMax: public Privates::WMax_Strategy_tag
		{
		public:
		    /** \brief Constructor
			 *
			 * Constructor that initializes random numbers generator. 
			 * Use one of generators from header random or Koala::StdRandGen for lower c++ standards.*/
		    GGWMax(RndGen& rg) : rgen(&rg) {}

			/* \brief Call function operator.
			 *
			 *  \param g the copy of considered graph with pointer of original vertices in info field.
			 *  \param vertTab the associative array that assigns weights to vertices.*/
			template< class GraphType, class VertContainer > typename GraphType::PVertex
				operator()( const GraphType &g, const VertContainer& vertTab );

            private: RndGen* rgen;
		};
	}
	/** \brief Maximal independent set heuristics (parametrized).
	 *
	 *  Class for max independent set.
	 *
	 *  Contains methods for two main templates: WMIN and WMAX.
	 *  It has to be initialized by a class containing a vertex
	 *  choice function called "choose".
	 *
	 *  \ingroup DMmis */
	template< class DefaultStructs > class MaxStableHeurPar
	{
	public:

		/** \brief Search maximal independent set (heuristic, WMin technique).
		 *
		 *  The method searches for maximal independent set using the following heuristic,
		 *  In each step (until the graph has no more vertices):
		 *   - chooses a vertex according to the choice function (Koala::MaxStableStrategy),
		 *   - adds this vertex to the independent set,
		 *   - removes the closed neighborhood of this vertex.
		 *
		 *  Since only heuristic is applied here the result may be suboptimal.
		 *  \param g the considered graph. Any type of graph is allowed. 
		 *   Mind that arcs are treated as undirected edges and vertices with loops may not belong to stable set. 
		 *  \param out the iterator to the container with the output set of vertices.
		 *  \param choose the strategy (\ref Koala::MaxStableStrategy) of choosing vertices (one in each step) .
		 *  \param vertTab the associative container that assigns weight to each vertex. blackHole possible if the funtcor is not using weights.
		 *  \return the number of vertices in the output set \a out.
		 *
		 *  [See example](examples/mis/example_mis_getWMin.html). */
		template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
			static unsigned getWMin( const GraphType &g, OutputIterator out, ChoiceFunction choose,
				const VertContainer & vertTab );

		/** \brief Search maximal independent set (heuristic, WMax technique).
		 * 
		 *  The method searches for maximal independent set using the following heuristic,
		 *  In each step (until the graph has no more edges):
		 *   - chooses a vertex according to the choice function (\ref Koala::MaxStableStrategy) ,
		 *   - removes the chosen vertex with adjacent edges.
		 *
		 *  Since only heuristic is applied here the result may be suboptimal.
		 *  The method outputs the remaining independent vertices.
		 *  \param g the considered graph. Any type of graph is allowed. 
		 *   Mind that arcs are treated as undirected edges and vertices with loops may not belong to stable set. 
		 *  \param out the iterator to the container with the output set of vertices.
		 *  \param choose the strategy (\ref Koala::MaxStableStrategy)  of choosing vertices (one in each step).
		 *  \param vertTab the associative container that assigns weight to each vertex. blackHole possible if the funtcor is not using weights.
		 *  \return the number of vertices in the output set \a out.
		 *
		 *  [See example](examples/mis/example_mis_getWMax.html). */
		template< class GraphType, class OutputIterator, class ChoiceFunction, class VertContainer >
			static unsigned getWMax( const GraphType &g, OutputIterator out, ChoiceFunction choose,
				const VertContainer &vertTab );

        /** \brief Test if stable
		 *
		 * Determinate if a set of vertices is independent.
		 *  \param g     - graph to process
		 *  \param first - first vertex from the potentially independent set
		 *  \param last  - past-the-last vertex from the potentially independent set
		 * \return true is the given set is independent, false otherwise. */
		template< class GraphType, typename Iterator >
			static bool test( const GraphType &g, Iterator first, Iterator last );

		/**\brief Test if max stable.
		 *
		 * Determinate if a set of vertices is maximal (in the sense of inclusion) i.e. if there is no vertices to add without spoiling stability. 
		 * If \a stabilitytest set true, the method also tests if the set is independent. 
		 * \param  g     - graph to process
		 * \param first - first vertex from the potentially independent set
		 * \param last  - past-the-last vertex from the potentially independent set.
		 * \param stabilitytest if set true the independence is tested. 
		 * \retrun true is the given set is maximal (in the sense of inclusion) independent, false otherwise.*/
		template< class GraphType, typename Iterator >
			static bool testMax( const GraphType &g, Iterator first, Iterator last, bool stabilitytest=true );

	private:
		/*
		 * Template:    WMIN
		 *
		 * In general in each step it (until the graph has no more vertices):
		 *   - chooses a vertex according to the choice function
		 *   - adds this vertex to the independent set
		 *   - removes the closed neighbourhood of this vertex
		 * After all it outputs the independent set which is maximal.
		 *
		 */
		template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
			static unsigned TemplateWMIN( GraphType &g, OutputIterator out, ChoiceFunction choose,
				const VertContainer &vertTab );

		/*
		 * Template:    WMAX
		 *
		 * In general in each step it (until the graph has no more edges):
		 *   - chooses a vertex according to the choice function
		 *   - removes the choosen vertex with adjacent edges
		 * All vertices not removed are outputed and this is an independent set wich is maximal.
		 *
		 */
		template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
			static unsigned TemplateWMAX( GraphType &g, OutputIterator out, ChoiceFunction choose,
				const VertContainer &vertTab );
	};

	/** \brief Maximal independent set heuristics (default).
	 *
	 *  Class for max independent set.
	 *
	 *  Contains methods for two main templates: WMIN and WMAX.
	 *  It has to be initialized by a class containing a vertex
	 *  choice function called "choose". \ref MaxStableStrategy
	 *
	 *  \ingroup DMmis */
	class MaxStableHeur: public MaxStableHeurPar< Koala::AlgsDefaultSettings > {};

    class LocalGrAdjMatrSettings : public AlgsDefaultSettings
	{
	public:

		template< class A, class B, EdgeType mask> class LocalGraph
		{
		public:
			typedef Graph< A,B,GrDefaultSettings< mask,true > > Type;
		};

	};

	/**\brief Maximum stable set exact algorithm.
	 *
	 * The class provides some non-polynomial time exact algorithm for stable (independent) set problem.
	 *
	 *  Based on: F. V. Fomin, F. Grandoni, D. Kratsch: Measure & conquer: A simple O(2^0.288n) independent set algorithm.
     *  ACM-SIAM Symposium on Discrete Algorithms (SODA), 18–25, 2006.*/
    template< class DefaultStructs > class MaxStablePar : private MaxStableHeurPar<DefaultStructs>
    {
      public:
        // znajduje najliczniejszy zbior niezalezny (procedura niewielomianowa). Jesli w trakcie szukania
        //wykryje, ze rozmiar minSize jest nieosioagalny, przerywa zwracajac -1
          /** \brief Calculate maximum independent set.
		   *
		   *  The method finds maximum independent set, however it is non-polynomial.
		   *  \param g  graph to process
		   *  \param out insert iterator to the output independent set
		   *  \param minSize the method stops and returns -1 if it recognizes that minSize is unachievable.
		   *  \return the number of vertices in the maximum independent set or -1 if the stability number is smaller then \a minSize.*/
		  template< class GraphType, class OutputIterator > static int
            findMax( GraphType & g, OutputIterator out, int minSize = 0);

        // znajduje zbior niezalezny mocy >= minSize (procedura niewielomianowa). Jesli w trakcie szukania
        //wykryje, ze rozmiar minSize jest nieosioagalny, przerywa zwracajac -1
          /** \brief Find independent set greater then \a minSize.
		   *
		   *  The method finds independent set greater then \a minSize, however it is non-polynomial.
		   *  \param g  graph to process
		   *  \param out insert iterator to the output independent set
		   *  \param minSize the method stops and returns -1 if it recognizes that minSize is unachievable.
		   *  \return the number of vertices in the found independent set or -1 if the stability number is smaller then \a minSize.*/
		  template< class GraphType, class OutputIterator > static int
            findSome( GraphType & g, OutputIterator out, int minSize);

            // testy na zb. niezalezny dziedziczone z MaxStableHeurPar
            using MaxStableHeurPar<DefaultStructs>::test;
            using MaxStableHeurPar<DefaultStructs>::testMax;

      private:
        /*
        * Maximum independent set - inner, recursive.
        */
        template< class GraphType, class OutputIterator > static int
            get( GraphType & g, OutputIterator out, int minSize, bool skipsearchiffound);
        template< class GraphType, class OutputIterator > static int
            getRecursive( GraphType &g, OutputIterator out, bool isConnectedComponent, bool outblackhole, int minSize, bool skipsearchiffound );
		template< class GraphType, class OutputIterator > static int
			getMirrors( const GraphType & g, typename GraphType::PVertex v, OutputIterator out);
        template< class GraphType, class InputIterator > static bool
			isClique( const GraphType &g, InputIterator beg, InputIterator end );
        template< class GraphType > static bool
			isDominated( const GraphType &g, typename GraphType::PVertex u, typename GraphType::PVertex v );
        template< class GraphType > static bool
			isFoldable( const GraphType &g, typename GraphType::PVertex v );
    };

    /**\brief Maximum stable set exact algorithm (default settings).
	 *
	 * The class provides some non-polynomial time exact algorithm for stable (independent) set problem.
	 *
	 *  Based on: F. V. Fomin, F. Grandoni, D. Kratsch: Measure & conquer: A simple O(2^0.288n) independent set algorithm.
     *  ACM-SIAM Symposium on Discrete Algorithms (SODA), 18–25, 2006.
	 * \sa MaxStablePar */
    //class MaxStable: public MaxStablePar< Koala::LocalGrAdjMatrSettings > {};
    class MaxStable: public MaxStablePar< Koala::AlgsDefaultSettings > {};


    /** \brief Maximum clique heuristics (parametrized).
	 *
	 *  Class provides heuristic approach to maximum clique problem.
	 *
	 *  Contains methods for two main templates: WMIN and WMAX. 
	 *  Both use algorithm for stable set delivered by MaxStablePar.
	 *  
	 *  \ingroup DMmis */
	template< class DefaultStructs > class MaxCliqueHeurPar
	{
	public:

		/** \brief Search maximum clique (heuristic, WMin technique).
		 *
		 *  The method searches for maximum clique by searching maximal independent set with method MaxStablePar::getWMin in negated graph. 
		 *  Since only heuristic is applied here the result may be suboptimal.
		 *  \param g the considered graph. Any type of graph is allowed. 
		 *   Mind that arcs are treated as undirected edges and loops are ignored. 
		 *  \param out the iterator to the container with the output set of vertices (clique).
		 *  \param choose the strategy (\ref Koala::MaxStableStrategy) of choosing vertices (one in each step) .
		 *  \param vertTab the associative container that assigns weight to each vertex. blackHole possible if the funtcor is not using weights.
		 *  \return the number of vertices in the output set \a out.*/
		template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
			static unsigned getWMin( const GraphType &g, OutputIterator out, ChoiceFunction choose,
				const VertContainer & vertTab );
		
		/** \brief Search maximum clique (heuristic, WMin technique).
		 *
		 *  The method searches for maximum clique by searching maximal independent set with method MaxStablePar::getWMax in negated graph. 
		 *  Since only heuristic is applied here the result may be suboptimal.
		 *  \param g the considered graph. Any type of graph is allowed. 
		 *   Mind that arcs are treated as undirected edges and loops are ignored. 
		 *  \param out the iterator to the container with the output set of vertices (clique).
		 *  \param choose the strategy (\ref Koala::MaxStableStrategy) of choosing vertices (one in each step) .
		 *  \param vertTab the associative container that assigns weight to each vertex. blackHole possible if the funtcor is not using weights.
		 *  \return the number of vertices in the output set \a out.*/
		template< class GraphType, class OutputIterator, class ChoiceFunction, class VertContainer >
			static unsigned getWMax( const GraphType &g, OutputIterator out, ChoiceFunction choose,
				const VertContainer &vertTab );

		/** \brief Test if clique
		 *
		 * The method tests if the vertices from container given by iterators \a first and \a last form a clique.
		 *  \param g     - graph to process
		 *  \param first - first vertex from the potential clique
		 *  \param last  - past-the-last vertex from the potential clique
		 *  \return true is the given set is a clique, false otherwise. */
		template< class GraphType, typename Iterator >
			static bool test( const GraphType &g, Iterator first, Iterator last );

		/** \brief Test if maximal clique.
		 *
		 *  The method tests if the vertices form container given by iterators \a first and \a last are maximal clique. 
		 *  I.e. if there exists a vertex outside the container that is incident with all vertices from container.
		 *  If \a stabilitytest is set false, it is assumed that vertices form container  form clique and the method only test if that set 
		 *  can be extended.
		 *  \param g     - graph to process
		 *  \param first - first vertex from the container 
		 *  \param last  - past-the-last element of the container.
		 *  \param stablilitytest - Boolean flag that decides if vertices from container are tested for being clique.
		 *  \return true is the given set is a maximal clique, false otherwise. */
		template< class GraphType, typename Iterator >
			static bool testMax( const GraphType &g, Iterator first, Iterator last, bool stabilitytest=true );

    protected:

        template< class Graph1, class Graph2 >
        static void copyneg(const Graph1& g, Graph2& h);

        template <class Cont>
        struct InfoPseudoAssoc {
            const Cont* cont;

            InfoPseudoAssoc(const Cont& arg) : cont(&arg) {}

            template <class Key>
            typename Cont::ValType operator[](Key key) const
            {
                return (*cont)[key->info];
            }
        };

	};

//	class MaxCliqueHeur: public MaxCliqueHeurPar< Koala::AlgsDefaultSettings > {};
	/** \brief Maximum clique heuristics (default algorithms settings).
	 *
	 *  Class provides heuristic approach to maximum clique problem.
	 *
	 *  Contains methods for two main templates: WMIN and WMAX.
	 *  Both use algorithm for stable set delivered by MaxStablePar.
	 *  
	 *  \ingroup DMmis */
	class MaxCliqueHeur: public MaxCliqueHeurPar< Koala::LocalGrAdjMatrSettings > {};

	/**\brief Maximum clique exact algorithm (parameterized).
	 *
	 * The class provides non-polynomial exact algorithm for maximum clique problem.  
	 * The used approach searches maximum stable set (See MaxStablePar) in negated graph.
	 * \sa MaxStablePar*/
    template< class DefaultStructs > class MaxCliquePar : private MaxCliqueHeurPar<DefaultStructs>
    {
      public:

		/**\brief Find maximum clique.
		 *
		 * The method determines maximum clique by searching independent set in negated graph. 
		 * Maximum stable set is found with method MaxStablePar::findMax. Mind that method is non-polynomial.
		 *  \param g  graph to process
		 *  \param out insert iterator to the output clique.
		 *  \param minSize the method stops and returns -1 if it recognizes that minSize is unachievable.
		 *  \return the number of vertices in the clique or -1 if there is no clique of size \a minSize.*/
		template< class GraphType, class OutputIterator > static int
            findMax( GraphType & g, OutputIterator out, int minSize = 0);

		/** \brief Find clique of size at least \a minSize.
		 *
		 *  The method finds clique not smaller then \a minSize, however it is non-polynomial.
		 *  \param g  graph to process
		 *  \param out insert iterator to the output clique.
		 *  \param minSize the method stops and returns -1 if it recognizes that minSize is unachievable.
		 *  \return the number of vertices in the found clique or -1 if there is no clique of size  \a minSize.*/
		template< class GraphType, class OutputIterator > static int
            findSome( GraphType & g, OutputIterator out, int minSize);

            using MaxCliqueHeurPar<DefaultStructs>::test;
            using MaxCliqueHeurPar<DefaultStructs>::testMax;
    };

    
	/**\brief Maximum clique exact algorithm (default algorithm settings).
	 *
	 * The class provides non-polynomial exact algorithm for maximum clique problem.  
	 * The used approach searches maximum stable set (See MaxStablePar) in negated graph.
	 * \sa MaxStablePar*/
	//class MaxClique: public MaxCliquePar< Koala::LocalGrAdjMatrSettings > {};
    class MaxClique: public MaxCliquePar< Koala::AlgsDefaultSettings > {};

    
    /** \brief Minimum vertex cover heuristics (parametrized).
	 *
	 *  Class provides heuristic approach for minimum vertex cover problem.
	 *
	 *  Contains methods for two main templates: WMIN and WMAX. 
	 *  Both use algorithm for stable set delivered by MaxStablePar.
	 *  
	 *  \ingroup DMmis */
	template< class DefaultStructs > class MinVertCoverHeurPar
	{
	public:
		
		/** \brief Search minimum vertex cover (heuristic, WMin technique).
		 *
		 *  The method searches for minimum vertex cover by searching maximal independent set with method MaxStablePar::getWMin.
		 *  Since only heuristic is applied here the result may be suboptimal.
		 *  \param g the considered graph. Any type of graph is allowed. 
		 *   Mind that arcs are treated as undirected edges and each vertex with loop is in the vertex cover. 
		 *  \param out the \wikipath{insert iterator} to the container with the output set of vertices (vertex cover.).
		 *  \param choose the strategy (\ref Koala::MaxStableStrategy) of choosing vertices (one in each step) .
		 *  \param vertTab the associative container that assigns weight to each vertex. blackHole possible if the funtcor is not using weights.
		 *  \return the number of vertices in the output set \a out.*/
		template< class GraphType, class ChoiceFunction, class OutputIterator, class VertContainer >
			static unsigned getWMin( const GraphType &g, OutputIterator out, ChoiceFunction choose,
				const VertContainer & vertTab );

		/** \brief Search minimum vertex cover (heuristic, WMin technique).
		 *
		 *  The method searches for minimum vertex cover by searching maximal independent set with method MaxStablePar::getWMax.
		 *  Since only heuristic is applied here the result may be suboptimal.
		 *  \param g the considered graph. Any type of graph is allowed. 
		 *   Mind that arcs are treated as undirected edges and each vertex with loop is in the vertex cover. 
		 *  \param out the \wikipath{insert iterator} to the container with the output set of vertices (vertex cover.).
		 *  \param choose the strategy (\ref Koala::MaxStableStrategy) of choosing vertices (one in each step) .
		 *  \param vertTab the associative container that assigns weight to each vertex. blackHole possible if the funtcor is not using weights.
		 *  \return the number of vertices in the output set \a out.*/
		template< class GraphType, class OutputIterator, class ChoiceFunction, class VertContainer >
			static unsigned getWMax( const GraphType &g, OutputIterator out, ChoiceFunction choose,
				const VertContainer &vertTab );

		/** \brief Test if vertex cover
		 *
		 * The method tests if the vertices from container given by iterators \a first and \a last form a vertex cover.
		 *  \param g     - graph to process
		 *  \param first - first vertex from the potential vertex cover
		 *  \param last  - past-the-last vertex from the potential vertex cover
		 *  \return true is the given set is a vertex cover, false otherwise. */
		template< class GraphType, typename Iterator >
			static bool test( const GraphType &g, Iterator first, Iterator last );

		/** \brief Test if minimal vertex cover.
		 *
		 *  The method tests if the vertices form container given by iterators \a first and \a last are minimal vertex cover. 
		 *  I.e. if there exists a vertex in the container that may be removed and the set still covers all the graph vertices.
		 *  If \a stabilitytest is set false, it is assumed that vertices from container cover all graph vertices and the method only test if that set 
		 *  can be reduced.
		 *  \param g     - graph to process
		 *  \param first - first vertex from the container 
		 *  \param last  - past-the-last element of the container.
		 *  \param stablilitytest - Boolean flag that decides if vertices form container are tested for covering all graph vertices.
		 *  \return true is the given set is a minimal (in sense of inclusion) vertex cover, false otherwise. */
		template< class GraphType, typename Iterator >
			static bool testMin( const GraphType &g, Iterator first, Iterator last, bool stabilitytest=true );

    protected:

        template< class GraphType, typename Iterator, typename IterOut >
        static int vertSetMinus(const GraphType &g, Iterator first, Iterator last,IterOut out);

	};

	//	class MinVertCoverHeur: public MinVertCoverHeurPar< Koala::AlgsDefaultSettings > {};
	/** \brief Minimum vertex cover heuristics (default algorithm settings).
	 *
	 *  Class provides heuristic approach for minimum vertex cover problem.
	 *
	 *  Contains methods for two main templates: WMIN and WMAX. 
	 *  Both use algorithm for stable set delivered by MaxStablePar.
	 *  
	 *  \ingroup DMmis */
	class MinVertCoverHeur: public MinVertCoverHeurPar< Koala::LocalGrAdjMatrSettings > {};

    /**\brief Minimum vertex cover exact algorithm (parameterized).
	 *
	 * The class provides non-polynomial exact algorithm for minimum vertex cover problem.  
	 * The used approach searches maximum stable set (See MaxStablePar).
	 * \sa MaxStablePar  
	 * \ingroup DMmis */
    template< class DefaultStructs > class MinVertCoverPar : private MinVertCoverHeurPar<DefaultStructs>
    {
      public:

        /**\brief Find minimum vertex cover (exact, non-polynomial).
		 *
		 * The method determines minimal vertex cover by searching independent set. 
		 * Maximum stable set is found with method MaxStablePar::findMax. Mind that method is non-polynomial.
		 *  \param g  graph to process
		 *  \param out \wikipath{insert iterator, output iterator} to the output clique.
		 *  \param maxSize the method stops and returns -1 if it recognizes that \a maxSize is unachievable.
		 *  \return the number of vertices in the minimum vertex cover or -1 if there is \a maxSize is unachievable.*/
		template< class GraphType, class OutputIterator > static int
            findMin( GraphType & g, OutputIterator out, int maxSize = std::numeric_limits< int >::max());

        /** \brief Find vertex cover of size at most \a maxSize.
		 *
		 *  The method finds a vertex cover not greater then \a maxSize, however it is non-polynomial.
		 *  \param g  graph to process
		 *  \param out \wikipath{insert iterator, output iterator} to the output clique.
		 *  \param maxSize the method stops and returns -1 if it recognizes that \a maxSize is unachievable.
		 *  \return the number of vertices in the found vertex cover or -1 if \a maxSize is unachievable.*/
		template< class GraphType, class OutputIterator > static int
            findSome( GraphType & g, OutputIterator out, int maxSize);

            using MinVertCoverHeurPar<DefaultStructs>::test;
            using MinVertCoverHeurPar<DefaultStructs>::testMin;
    };

    /**\brief Minimum vertex cover exact algorithm (default algorithm settings).
	 *
	 * The class provides non-polynomial exact algorithm for minimum vertex cover problem.  
	 * The used approach searches maximum stable set (See MaxStablePar).
	 * \sa MaxStablePar  
	 * \ingroup DMmis */
    //class MinVertCover: public MinVertCoverPar< Koala::LocalGrAdjMatrSettings > {};
    class MinVertCover: public MinVertCoverPar< Koala::AlgsDefaultSettings > {};

#include "mis.hpp"
}

#endif
