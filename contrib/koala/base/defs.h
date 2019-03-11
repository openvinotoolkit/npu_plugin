#ifndef KOALA_DEFS_H
#define KOALA_DEFS_H

/** \file defs.h
 *  \brief Collection of auxiliary classes (included automatiaclly).*/

#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <stack>
#include <utility>

#include "exception.h"
#include "../container/assoctab.h"
#include "../container/hashcont.h"
#include "../container/heap.h"
#include "../container/localarray.h"
#include "../container/simple.h"
#include "../container/set.h"

namespace Koala
{
	/** \brief Edge types.
	 *
	 *  Type used for variables storing basic information about edge type. Applied to masks with bit meaning as follows:
	 *  - Loop       = 0x1  - edges with only one vertex.
	 *  - Undirected = 0x2  - undirected edges .
	 *  - Directed   = 0xC  - arcs - directed edge.
	 *  \ingroup DMgraph */
	typedef unsigned char EdgeType;

	/** \brief Edge direction.
	 *
	 *  The type used for variables storing the information about direction of edge.
	 *  Variables of this type can be used as masks.\n
	 *  Bits meaning:
	 *  -  EdLoop   = 0x01  - edge with only one vertex
	 *  -  EdUndir  = 0x02  - undirected edge
	 *  -  EdDirIn  = 0x04  - for directed edge in direction
	 *  -  EdDirOut = 0x08  - for directed edge out direction
	 *  -  EdAll    = 0x0F  - for all above options
	 *   \ingroup DMgraph */
	typedef unsigned char EdgeDirection;

	static const EdgeDirection EdNone   = 0x00;
	static const EdgeDirection EdLoop   = 0x01;
	static const EdgeDirection EdUndir  = 0x02;
	static const EdgeDirection EdDirIn  = 0x04;
	static const EdgeDirection EdDirOut = 0x08;
	static const EdgeDirection EdAll    = 0x0F;

	static const EdgeType Detached   = 0x0;
	static const EdgeType Loop       = 0x1;
	static const EdgeType Undirected = 0x2;
	static const EdgeType Directed   = 0xC;

	/** \brief Structures for empty vertex info.
	 *
	 *  The empty structure often used as default value for info attributes in vertices.
	 *  \ingroup DMdef*/
	struct EmptyVertInfo { } ;
	/** \brief Structures for empty edge info.
	 *
	 *  The empty structure often used as default value for info attributes in edges.
	 *  \ingroup DMdef*/
	struct EmptyEdgeInfo { } ;

	template< class VertInfo, class EdgeInfo, class Settings > class Graph;
	template< class VertInfo, class EdgeInfo, class Settings > class Vertex;
	template< class VertInfo, class EdgeInfo, class Settings > class Edge;
	template< EdgeType mask, bool matr > class GrDefaultSettings;


	/** \brief Numeric types specialization.
	 * 
	 *  Class allows to choose own numeric types for data in internal Koala algorithms.
	 *  This is a Koala version of std::numeric_limits.*/
    template< class T > class NumberTypeBounds
    {
    public:
		/**\brief Get type maximal value.*/
        static T plusInfty()
            { return std::numeric_limits< T >::max(); }
		/**\brief Test if type maximal value.*/
		static bool isPlusInfty(T arg)
            { return arg == plusInfty(); }
		/**\brief */
		static T minusInfty()
            { return std::numeric_limits< T >::min(); }
		/**\brief */
		static bool isMinusInfty(T arg)
            { return arg == minusInfty(); }
		/**\brief Get zero value.*/
		static T zero()
            { return (T)0; }
		/**\brief */
		static T one()
            { return (T)1; }
		/**\brief */
		static bool isZero(T arg)
            { return arg == zero(); }
    };

	/** \brief Default algorithms settings.
	 *
	 *  This is an useful plug-in that allows to parameterize some algorithms in this library with default values.
	 *  An usual class is declared  <tt> SomeNamePar<class DefaultStructs> </tt> In most cases there is also
	 *  a class \a SomeName which is achieved from the original one by simply setting <tt>DefaultStructs = AlgsDefaultSettings</tt>
	 *  \ingroup DMdef */
	class AlgsDefaultSettings
	{
	public:
		/** \brief Type of associative container
		 *
		 *  The class is most often used to assign some values (colors weights priority etc.) to vertices (PVertex) or edges (PEdge).
		 *  \tparam A key type.
		 *  \tparam B mapped value type. */
		template< class A, class B > class AssocCont
		{
		public:
			/**\brief Associative container type.
			 *
			 * Define this type as other associative container in order to introduce changes.
			 * Exemplary other possibilities:
			 * - typedef AssocTable < BiDiHashMap<A,B> > Type;
			 * - typedef AssocTable < HashMap<A,B> > Type;
			 * - typedef AssocTable < std::map<A,B> > Type;	 */
			typedef AssocArray< A,B > Type;
		};

		/** \brief Two dimensional associative array.
		 *
		 *  \tparam A the key type.
		 *  \tparam B the mapped value type.
		 *  \tparam type the kind of associative matrix (Koala::AssocMatrixType). */
		template< class A, class B, AssocMatrixType type > class TwoDimAssocCont
		{
		public:
			typedef SimpleAssocMatrix< A,B,type > Type;/**<\brief Define own if intend to change.*/

			// Exemplary usage. Other possibilities:

            //  typedef SimpleAssocMatrix<A,B,type,std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<A,B>::BlockType> >,Privates::PseudoAssocArray<A,int,AssocTable<BiDiHashMap<A,int> > > > Type;
            //  typedef SimpleAssocMatrix<A,B,type,std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<A,B>::BlockType> >,Privates::PseudoAssocArray<A,int,AssocTable<HashMap<A,int> > > > Type;
            // possible, but access time is logarithmic
            //  typedef SimpleAssocMatrix<A,B,type,std::vector< std::vector<typename Privates::SimpleAssocMatrixInternalTypes<A,B>::BlockType> >,Privates::PseudoAssocArray<A,int,AssocTable<std::map<A,int> > > > Type;

			// typedef AssocMatrix< A,B,type > Type;

			//  typedef AssocMatrix<A,B,type,std::vector< Privates::BlockOfAssocMatrix<B> >,Privates::PseudoAssocArray<A,int,AssocTable<BiDiHashMap<A,int> > > > Type;
            //  typedef AssocMatrix<A,B,type,std::vector< Privates::BlockOfAssocMatrix<B> >,Privates::PseudoAssocArray<A,int,AssocTable<HashMap<A,int> > > > Type;
            // possible but not recommended - access time is logarithmic
            //  typedef AssocMatrix<A,B,type,std::vector< Privates::BlockOfAssocMatrix<B> >,Privates::PseudoAssocArray<A,int,AssocTable<std::map<A,int> > > > Type;

            // wrappers for 2-dimentional maps
            //  typedef  Assoc2DimTable< type, std::map<std::pair<A,A>, B > > Type;
            //  typedef  Assoc2DimTable< type, BiDiHashMap<std::pair<A,A>, B > > Type;
            //  typedef  Assoc2DimTable< type, HashMap<std::pair<A,A>, B > > Type;
		};

		/** \brief Heap container.
		 *
		 *  \tparam key the key class.
		 *  \tparam Compare the comparison object function.*/
		template< class Key, class Compare = std::less< Key > >
			class HeapCont
		{
		public:
			typedef FibonHeap< Key,Compare > Type;/**<\brief Define own if intend to change.*/
            typedef FibonHeapNode< Key > NodeType;/**<\brief Define own if intend to change.*/

			//Exemplary usage. Other possibilities:
            //typedef BinomHeap< Key,Compare > Type;/**<\brief Define own if intend to change.*/
                    //typedef BinomHeapNode< Key > NodeType;/**<\brief Define own if intend to change.*/

            //typedef PairHeap< Key,Compare > Type;/**<\brief Define own if intend to change.*/
                    //typedef PairHeapNode< Key > NodeType;/**<\brief Define own if intend to change.*/
		};

		/** \brief Auxiliary graph.
		 *  
		 *  The structure is used for internal purposes in various procedures.
		 *  \tparam A VertInfo type.
		 *  \tparam B EdgeInfo type
		 *  \tparam mask Edge types in graph. \wikipath{EdgeType,See wiki page for EdgeType.}*/
		template< class A, class B, EdgeType mask> class LocalGraph
		{
		public:
			/** \brief Define in order to set own */
			typedef Graph< A,B,GrDefaultSettings< mask,false > > Type;
		};

		enum { ReserveOutAssocCont /**< \brief Should the out associative container be allocated at the beginning?*/ = true };

		/**  \brief Container sorting algorithm
		 *
		 * The functions sorts elements in container given by iterators first last. Define own in order to introduce changes.
		 * \tparam Iterator the iterator class for container.
		 * \param first the iterator of the first element in container.
		 * \param last the iterator of the past the last element in container.*/
		template< class Iterator > static void sort( Iterator first, Iterator last )
		{
            std::make_heap( first,last );
            std::sort_heap( first,last );
        }
		/**  \brief Container sorting algorithm
		*
		* The functions sorts elements in container given by iterators first last. The object function comp is used for comparing elements.
		* Define own in order to introduce changes.
		* \tparam Iterator the iterator class for container.
		* \param first the iterator of the first element in container.
		* \param last the iterator of the past the last element in container.
		* \tparam Comp the class of comparison object function. The functor should give strict weak order.  Similar to one in std::sort.
		* \param comp the comparison object function.*/
		template< class Iterator, class Comp > static void sort( Iterator first, Iterator last, Comp comp )
		{
            std::make_heap( first,last,comp );
            std::sort_heap( first,last,comp );
        }

        //other possibility, taken from std::sort
//		template< class Iterator > static void sort( Iterator first, Iterator last )
//		{
//            std::sort( first,last );
//        }
//		template< class Iterator, class Comp > static void sort( Iterator first, Iterator last, Comp comp )
//		{
//            std::sort( first,last,comp );
//        }

	};


	/** \brief Constant functor.
	 *
	 *  The default function object can be used if method requires the object function, generating for example
	 *   edge info, but the user does not need to specify it. The functor works with 0 to 6 arguments and always
	 *   returns the value prespecified in constructor.
	 *   \tparam T The type of returned object, the type have minimal requirements similar to STL objects. I.e. it must implement:
	 *   - empty constructor
	 *   - copy constructor
	 *   - destructor
	 *   - operator=
	 *  \ingroup DMdef*/
	template< class T > class ConstFunctor
	{
		const T val;

	public:
		/** \brief Constructor.
		 *
		 *  Defines the value return by calls of functor. */
		ConstFunctor( const T &aval = T() ): val( aval ) { }

		/** \brief No arguments functor. */
		inline T operator()()
			{ return val; }

		/** \brief Single argument functor. */
		template< class A > T operator()(const A&)
				{ return val; }
		/** \brief Two arguments functor. */
		template< class A, class B > T operator()(const A&, const B& )
				{ return val; }
		/** \brief Three arguments functor. */
		template< class A, class B, class C > T operator()( const A&,const B&,const C& )
				{ return val; }
		/** \brief Four arguments functor. */
		template< class A, class B, class C, class D > T operator()(  const A&,const B&,const C&,const D& )
				{ return val; }
		/** \brief Five arguments functor. */
		template< class A, class B, class C,class D, class E > T operator()( const A&,const B&,const C&,const D&,const E& )
				{ return val; }
		/** \brief Six arguments functor. */
		template< class A, class B, class C,class D, class E, class F > T operator()( const A&,const B&,const C&,const D&,const E&,const F& )
				{ return val; }
	};

	/** \brief Generating function for constant functor.
	 *
	 * \related ConstFuctor
	 *
	 *  \ingroup DMdef */
	template< class T > ConstFunctor< T > constFun( const T &a = T() )
				{ return ConstFunctor< T >( a ); }

	/** \brief Black hole.
	 *
	 *  Sometimes method does more than the user wants. Than the class succor. It can supersede \wikipath{insert iterators, Output_iterator}
	 *  of containers or associative tables as long as the result is never used by the user.
	 *  \ingroup DMdef */
	struct BlackHole: public std::iterator< std::output_iterator_tag,void,void,void,void >
	{
		template< class T > BlackHole &operator=( T )
			{ return *this; }

		BlackHole &operator*()
			{ return *this; }
		BlackHole &operator++()
			{ return *this; }
		BlackHole operator++( int )
			{ return *this; }
        bool operator==(BlackHole)
            { return true; }
        bool operator!=(BlackHole)
            { return false; }

		BlackHole()
			{}

		// BlackHole can be plugged as an associative array that we don't need but it necessary to compile
		// Below methods should never be used.
		template< class T > inline BlackHole &operator[]( T );
		template< class T, class R > inline BlackHole &operator()( T,R );

		// BlackHole can change type into any type
		template< class T > inline operator T();

		template< class T > inline bool hasKey(T) const;
		inline BlackHole  firstKey() const;
		inline BlackHole  lastKey() const;
		template< class T > inline BlackHole nextKey(T) const;
		template< class T > inline BlackHole prevKey(T) const;
		template< class T > inline int getKeys(T) const;
		void reserve( int )
			{ }
		bool empty() const
			{ return true; }
		bool operator!() const
			{ return true; }
		inline unsigned size() const;
		inline int capacity() const;
		template< class T > bool delKey(T)
			{ return false; };
		void clear()
			{ }

	};

	/** \def blackHole 
	 *  \brief BlackHole macro.
	 *
	 *  The macro inserts BlackHole object \wikipath{blackHole}.
	 *  \related BlackHole 
	 *  \ingroup DMdef */
	#define blackHole ((*((Koala::BlackHole*)( &std::cout ))))

	/** \brief Test if black hole.
	 *
	 *  The	method tests if type \a T is BlackHole. Although it always returns false,
	 *  there is a specialization of it available for BlackHole type, which returns true.
	 *  \return false unless the specialization for BlackHole is called.
	 *  \related BlackHole
	 *  \ingroup DMdef */
	template< class T > bool isBlackHole( const T & )
		{ return false; }
	/* \brief Test if black hole.
	 *
	 *  \return true if the tested type is BlackHole.
	 *  \related BlackHole
	 *  \ingroup DMdef */
	inline bool isBlackHole( const BlackHole & )
		{ return true; }

	/** \brief Switch blackHole to local container
	 *
	 *  If Cont1 is BlackHole method get delivers container of type Cont2 otherwise it returns object of type Cont1.
	 *  \ingroup DMdef */
	template< class Cont1, class Cont2 > struct BlackHoleSwitch
	{
		// Type of container we use
		typedef Cont1 Type;

		/** \brief Get container.
		 *
		 *
		 */
		static Cont1 &get( Cont1 &a, Cont2 &b )
			{ return a; }
		static const Cont1 &get(const Cont1 &a, const Cont2 &b )
			{  return a; }
	};

	template< class Cont2 > struct BlackHoleSwitch< BlackHole,Cont2 >
	{
		typedef Cont2 Type;

		static Cont2 &get( BlackHole &a, Cont2 &b )
			{ return b; }
		static const Cont2 &get(const BlackHole &a,const Cont2 &b )
			{  return b; }
	};


	// Choosers return true/false (by operator()) for vertices/edges. They cane be used to select subgraphs and should be
    // created by their generating functions.

	// Universal choosers - for edges and vertices

	/** \brief Fixed chooser
	 *
	 *  Function object class that always returns value true or false, depending on the value set in constructor.
	 *  This chooser should be used whenever each (or none) object is to be chosen. The chooser works with both edges and vertices.
	 *  \wikipath{Chooser, See for more details about choosers.}
	 *  \ingroup DMchooser */
	struct BoolChooser
	{
		bool val;/**<\brief Logic value fixed in constructor returned by each call of function object. */

		/** \brief Chooser obligatory type.
		 *
		 *  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef BoolChooser ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Assigns value to field \a val. */
		BoolChooser( bool arg = false ): val( arg ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning Boolean value \a val (the same in each call of operator).
		 *  \param elem the considered object.
		 *  \param gr reference to considered graph (not used in this chooser).
		 *  \return the value \a val. */
		template< class Elem, class Graph > bool operator()( Elem *elem, const Graph &gr ) const { return val; }
	};


	// TODO: sprawdzic, czy rozne przeciazenia funkcji tworzacych nie wywoluja niejednoznacznosci w rozstrzyganiu przeciazen
	/** \brief Generating  function of fixed chooser (BoolChooser).
	 *
	 *  The function generates BoolChooser function object, that returns value \a arg for each call of operator is called.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \param arg Boolean that will be returned by each call of the chooser.
	 *  \ingroup DMchooser
	 *  \related BoolChooser */
	inline BoolChooser stdChoose( bool arg ) { return BoolChooser( arg ); }

	/** \brief Value chooser
	 *
	 *  Function object that compares the fixed value \a val defined in constructor to the one given by parameter \a elem in calls of overloaded operator().
	 *  Chooser should be used whenever simple comparison to fixed value is necessary,
	 *   for example only one object is to be chosen. The chooser works with both edges and vertices.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Elem class of compared value.
	 *  \ingroup DMchooser */
	template< class Elem > struct ValChooser
	{
		Elem val; /**< \brief value fixed in constructor */

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef ValChooser< Elem > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Assigns value to variable \a val. */
		ValChooser( Elem arg ): val( arg ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true if objects \a elem and \a val match and false otherwise.
		 *  \param elem element compared to \a val.
		 *  \param gr reference to considered graph (not used in this chooser).
		 *  \return true if \a elem equals to \a val false otherwise.	 */
		template< class Graph > bool operator()( Elem elem, const Graph &gr) const { return elem == val; }
	};

	/** \brief Generating function of value chooser (ValCooser).
	 *
	 *  The function generates ValChooser function object that tests whether checked element equals \a arg. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Elem the type of tested element, possibly deduced from \a arg
	 *  \param arg the value for which chooser returns true.
	 *  \ingroup DMchooser
	 *  \related ValChooser */
	template< class Elem > ValChooser< Elem > stdValChoose( Elem arg ) { return ValChooser< Elem >( arg ); }

	/** \brief Set chooser
	 *
	 *  Function object that checks if \a elem (the parameter in call of overloaded operator()) belongs to the set defined in constructor.
	 *  Chooser should be used whenever elements from given set are to be chosen. It works with both edges and vertices.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Elem the class of compared value.
	 *  \ingroup DMchooser*/
	template< class Elem > struct SetChooser
	{
		Koala::Set< Elem * > set;/**< \brief Fixed in constructor set of elements of \a Elem type. */

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef SetChooser< Elem > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Determines the set of elements. */
		SetChooser( const Koala::Set< Elem * > &arg = Koala::Set< Elem * >() ): set( arg ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true if object \a elem belongs to the set defined in constructor and false otherwise.
		 *  \param elem the checked element.
		 *  \param gr reference to considered graph (not used in this chooser).
		 *  \return true if \a elem belongs to the \a set, false otherwise.	 */
		template< class Graph > bool operator()( Elem *elem, const Graph &gr ) const { return set.isElement( elem ); }
	};
	/** \brief Generating function of set chooser (SetChooser).
	 *
	 *  The function generates SetChooser object function that for call of function call operator returns true as long as
	 *  element \a arg belong to Set. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Elem the type of tested elements.
	 *  \param arg the set of elements (pointers) for which chooser function call returns true.
	 *  \related SetChooser
	 *  \ingroup DMchooser*/
	template< class Elem >
		SetChooser< Elem > stdChoose( Koala::Set< Elem * > arg ) { return SetChooser< Elem >( arg ); }

	/** \brief Container element chooser
	 *
	 *  Function object that checks if parameter \a elem in call of overloaded operator() belongs to the container defined in constructor.
	 *  The container is given be iterators to the first and to the past-the-last element. It should like STL containers serve std::find algorithm.
	 *  Since the container is not copied it is users prerogative to keep iterators up to date.
	 *  The chooser should be used whenever elements from given container should be considered.
	 *  It works with both edges and vertices.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Iter the iterator class used to access a container.
	 *  \ingroup DMchooser*/
	template< class Iter > struct ContainerChooser
	{
		Iter begin; /**< \brief iterator to the first element of container */
		Iter end; /**< \brief iterator to the past-the-end element of container */

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef ContainerChooser< Iter > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Assigns value to iterators defining the container.
		 *  \param abegin the iterator to the first element in the container with elements.
		 *  \param aend past-the-end element of the container with elements.*/
		ContainerChooser( Iter abegin = Iter(), Iter aend = Iter() ): begin( abegin ), end( aend ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true if objects \a elem belongs to the container defined in constructor and false otherwise.
		 *  \param elem the checked element.
		 *  \param gr the considered graph (not used in this chooser).
		 *  \return true if \a elem belongs to the container, false otherwise. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph & ) const { return std::find( begin,end,elem ) != end; }
	};

	/** \brief Generating function of container element chooser (ContainerChooser).
	 *
	 *  The function generates chooser object function ContainerChooser that chooses only elements from container
	 *  given by iterators \a begin and \a end.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Iter the type of container iterator.
	 *  \param begin the iterator to the first element of container.
	 *  \param end the iterator to the past-the-end element of container.
	 *  \ingroup DMchooser
	 *  \related ContainerChooser*/
	template< class Iter >
		ContainerChooser< Iter > stdChoose( Iter begin, Iter end ) { return ContainerChooser< Iter >( begin,end ); }

	/** \brief Function object chooser.
	 *
	 *  Wraps self-made function object.
	 *  Function object should take two parameters chosen element and considered graph. It should return value convertible to Boolean type.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Obj the function object class used to choose entities.
	 *  \ingroup DMchooser */
	template< class Obj > struct ObjChooser
	{
		mutable Obj functor; /**< \brief Function object defined in constructor.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef ObjChooser< Obj > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Assigns function object to functor. */
		ObjChooser( Obj arg = Obj() ): functor( arg ) { }

		/** \brief Overloaded operator()
		 *
		 *  The function call operator returning Boolean value the same as the function object defined in the constructor.
		 *  \param elem the checked element.
		 *  \param graph the considered graph.
		 *  \return the same vales (casted on bool) as functor \a arg. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph &graph ) const { return (bool)functor( elem,graph ); }
	};

	/** \brief Generating function of function object chooser (ObjChooser).
	 *
	 *  The function wraps object function \a arg and generates chooser object function.
	 *  \tparam Obj the type of wrapped object function.
	 *  \param arg the wrapped object function.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser
	 *  \related ObjChooser*/
	template< class Obj > ObjChooser< Obj > stdFChoose( Obj arg ) { return ObjChooser< Obj >( arg ); }

	/** \brief Info field value chooser
	 *
	 *  Function object that checks if the attribute \a val matches an element of info object field pointed by \a wsk.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \ingroup DMchooser */
	template< class Info, class T > struct FieldValChooser
	{
		T Info:: *wsk; /**<\brief Pointer to member.*/
		T val; /**< \brief Desired value. */

		/** \brief Chooser obligatory type.
		 *
		 *  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef FieldValChooser< Info, T > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Assigns values to \a val and pointer to member \a wsk i.e. decides which attribute of info are to be checked and what value are they to be equal to.
		 *  \param arg pointer to member in Info object.
		 *  \param arg2 the desired value of tested field.*/
		FieldValChooser( T Info:: *arg = 0, T arg2 = T() ): wsk( arg ), val( arg2 ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true as long as \a elem->info attribute pointed by \a wsk matches the value \a val.
		 *  \param elem the checked element.
		 *  \param graph the considered graph (not used by this chooser).
		 *  \return true if \a val equals pointed by \a wsk member of info in object \a elem, false otherwise. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph &graph ) const { return elem->info.*wsk == val; }
	};

	/** \brief Generating function of field value chooser (FieldValChooser).
	 *
	 *  The function generates chooser elements in which info object filed pointed by \a wsk math \a arg.
	 *  \tparam Info the type of Info object.
	 *  \tparam T the type tested member.
	 *  \param wsk pointer to tested member in Info object.
	 *  \param arg desired value of tested field.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \related FieldValChooser
	 *  \ingroup DMchooser*/
	template< class Info, class T >
		FieldValChooser< Info,T > fieldChoose( T Info:: *wsk, T arg ) { return FieldValChooser< Info,T >( wsk,arg ); }

	/** \brief Less info field value chooser
	 *
	 *  Function object that chooses elements for which the attribute pointed by \a wsk in info object is
	 *  lower (operator< on type T is used) then  \a val (set in constructor).
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \ingroup DMchooser */
	template< class Info, class T > struct FieldValChooserL
	{
		T Info:: *wsk; /**< \brief Pointer to member.*/
		T val; /**< \brief Value to compare to fixed in constructor */

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef FieldValChooserL< Info, T > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Assigns values to \a val which is compared to proper attribute in info object
		 *  and sets pointer to member \a wsk i.e. decides which members of info is to be considered.
		 *  \param arg pointer to member
		 *  \param arg2 the compared value*/
		FieldValChooserL( T Info:: *arg = 0, T arg2 = T() ): wsk( arg ), val( arg2 ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true as long as field of \a elem pointed by pointer to member \a wsk is less than \a val.
		 *  \param elem the checked element.
		 *  \param graph the reference to considered graph.
		 *  \return true if \a val is greater than pointed by the \a wsk member of info in the object \a elem, false otherwise. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph &graph ) const { return elem->info.*wsk < val; }
	};

	/** \brief Generating function of less field value chooser (FieldValChooserL).
	 *
	 *  The function generates FieldValChooserL chooser object function that returns true if and only if field pointed by \a wsk in \a Info object
	 *  is lower then \a arg.
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \param wsk pointer to tested member in \a Info object.
	 *  \param arg the desired value of approved elements.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser*/
	template< class Info, class T > FieldValChooserL< Info,T >
		fieldChooseL( T Info:: *wsk, T arg ) { return FieldValChooserL< Info,T >( wsk,arg ); }

	/** \brief Greater info field value chooser
	 *
	 *  Function object that chooses elements for which the attribute pointed by \a wsk in info object is
	 *  greater (operator< on type T is used) then \a val (set in constructor).
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \ingroup DMchooser */
	template< class Info, class T > struct FieldValChooserG
	{
		T Info:: *wsk;/**< \brief Pointer to member.*/
		T val;/**< \brief Value to compare to fixed in constructor */

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef FieldValChooserG< Info, T > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Assigns values to \a val that is compared to proper info attribute
		 *  and pointer to member \a wsk that chooses the compared attribute.
		 *  \param arg pointer to member
		 *  \param arg2 the compared value */
		FieldValChooserG( T Info:: *arg = 0, T arg2 = T() ): wsk( arg ), val( arg2 ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true as long as the field of \a elem
		 *  pointed by \a wsk is greater than \a val.
		 *  \param elem pointer to checked object.
		 *  \param graph the considered graph.
		 *  \return true if \a val equals pointed by \a wsk member of info in object \a elem, false otherwise. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph &graph ) const { return elem->info.*wsk > val; }
	};

	/** \brief Generating function of greater field value chooser (FieldValChooserG).
	 *
	 *  The function generates FieldValChooserG chooser object function that returns true if and only if field pointed by \a wsk in \a Info object
	 *  greater than \a arg. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \param wsk pointer to tested member in \a Info object.
	 *  \param arg the desired value of approved elements.
	 *  \related FieldValChooserG
	 *  \ingroup DMchooser*/
	template< class Info, class T > FieldValChooserG< Info,T >

		fieldChooseG( T Info:: *wsk, T arg ) { return FieldValChooserG< Info,T >( wsk,arg ); }

	/** \brief Boolean info field chooser
	 *
	 *  Function object that checks if certain filed of element \a elem from
	 *  overloaded \p operator() is convertible to true value.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info class of object info.
	 *  \tparam T type of compared field.
	 *  \ingroup DMchooser */
	template< class Info, class T > struct FieldBoolChooser
	{
		T Info:: *wsk;/**< \brief Pointer to tested member.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef FieldBoolChooser< Info,T > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Decides which members of info are to be checked.
		 *  \param arg pointer to tested member.*/
		FieldBoolChooser( T Info::*arg = 0): wsk( arg ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true as long as filed pointed by \a wsk in \a elem info object is convertible to true.
		 *  \param elem the checked object.
		 *  \param graph the considered graph.
		 *  \return true if pointed by \a wsk member of info in object \a elem is true.
		 *  (convertible to true value), false otherwise. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph &graph ) const { return bool( elem->info.*wsk ); }
	};

	/** \brief Generating function of bool field chooser (FielBoolChooser).
	 *
	 *  The function generates function object FielBoolChooser that tests if info filed pointed by \a wsk is convertible to true.
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \param wsk pointer to tested member in \a Info object.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \related FieldBoolChooser
	 *  \ingroup DMchooser*/
	template< class Info, class T >
		FieldBoolChooser< Info,T > fieldChoose( T Info:: *wsk ) { return FieldBoolChooser< Info,T >(wsk); }

	/** \brief Functor wrapper.
	 *
	 *  Function object that checks if the given \a functor returns value convertible to true for a certain (pointed by \a wsk) field of info object.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \tparam Obj the function object class that provides function testing the field. The object function must work with type of pointed member.
	 *  \ingroup DMchooser */
	template< class Info, class T, class Obj > struct FieldObjChooser
	{
		T Info:: *wsk; /**< \brief the pointer to tested member.*/
		mutable Obj functor; /**< \brief functor testing the member. */

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef FieldObjChooser< Info, T, Obj > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  \param awsk to pointer to tested member
		 *  \param afun the wrapped object function */
		FieldObjChooser( T Info:: *awsk = 0, Obj afun = Obj() ): wsk( awsk ), functor( afun ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if wrapped function object returns true for filed pointed by \a wsk in \a elem info object.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return the same value as wrapped objet. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph &graph ) const { return (bool)functor( elem->info.*wsk ); }
	};

	/** \brief Generating  function of FieldObjChooser.
	 *
	 *  The function generates function object FieldObjChooser that tests if info filed pointed by \a wsk is accepted by functor \a obj.
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \tparam Obj the type of wrapped function object.
	 *  \param wsk pointer to tested member in \a Info object.
	 *  \param obj the wrapped function object. Should implement overloaded function call operator for single parameter of pointed member type
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \related FieldObjChooser
	 *  \ingroup DMchooser*/
	template< class Info, class T, class Obj > FieldObjChooser< Info,T,Obj >
		fieldFChoose( T Info::*wsk, Obj obj ) { return FieldObjChooser< Info,T,Obj >( wsk,obj ); }

	/** \brief Info field value belongs to set chooser.
	 *
	 *  Function object that checks if prespecified attribute of info object belongs to the set defined in constructor.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \tparam Z the type of objects in set.
	 *  \ingroup DMchooser */
	template< class Info, class T, class Z > struct FieldSetChooser
	{
		T Info:: *wsk; /**<\brief pointer to tested member in info object*/
		Koala::Set< Z > set;/**<\brief the set of allowed values*/

		/** \brief Chooser obligatory type.
		 *
		 *  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef FieldSetChooser< Info, T, Z > ChoosersSelfType;

		/**\brief Constructor
		 *
		 * The constructor sets pointer to tested member \a awsk and initializes set of allowed values \a aset.*/
		FieldSetChooser( T Info:: *awsk = 0, const Koala::Set< Z > &aset = Koala::Set< Z >() ):
			wsk( awsk ), set( aset ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true if and only if value pointed by \a wsk in info attribute of \a elem belongs to \a set.
		 *  \param elem the checked object.
		 *  \param graph the considered graph.
		 *  \return true if and only if pointed attribute in \a elem info object belong to \a set. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph &graph ) const { return set.isElement( elem->info.*wsk ); }
	};

	/** \brief Generating  function of FieldSetChooser.
	 *
	 *  FieldSetChooser function object is generated. The functor test if field pointed by \a wsk
	 *  in info object of tested element belongs to \a set.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \tparam Z the type of objects in set.
	 *  \param wsk pointer to tested member.
	 *  \param set the set of approved values.
	 *  \related FieldSetChooser
	 *  \ingroup DMchooser*/
	template< class Info, class T, class Z > FieldSetChooser< Info,T,Z >
		fieldChoose( T Info::*wsk, Koala::Set< Z > set ) { return FieldSetChooser< Info,T,Z >( wsk,set ); }

	/** \brief Info field value belong to container chooser.
	 *
	 *  Function object that checks if certain field of info object belongs to the container given by iterators.
	 *  Container should behave like stl one, function std::find must be applicable.
	 *  The container is not copied and it is users prerogative to keep iterators valid.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \tparam Iter the type of iterator for container with tested elements.
	 *  \ingroup DMchooser */
	template< class Info, class T, class Iter > struct FieldContainerChooser
	{
		T Info:: *wsk;/**< \brief Pointer to tested member.*/
		Iter begin/**< \brief The iterator to the first element of container.*/, end;/**< \brief The iterator to the past-the-end element of container.*/

		/** \brief Chooser obligatory type.
		 *
		 *  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef FieldContainerChooser< Info, T, Iter > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The constructor sets pointer to tested member and determines the container with compared elements.
		 *  \param awsk the pointer to tested member
		 *  \param a The iterator to the first element of container.
		 *  \param b The iterator to the past-the-end element of container.*/
		FieldContainerChooser( T Info:: *awsk = 0, Iter a = Iter(), Iter b = Iter() ):
			wsk( awsk ), begin( a ), end( b ) {}

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if value pointed by \a wsk in info attribute of \a elem belongs to container.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if pointed attribute in \a elem info object belong to the container. */
		template< class Elem, class Graph > bool
			operator()( Elem *elem, const Graph &graph ) const { return std::find( begin,end,elem->info.*wsk ) != end; }
	};

	/** \brief Generating  function of FielContainerChooser.
	 *
	 *  FieldContainerChooser function object is generated. The functor test if field pointed by \a wsk
	 *  in info object of tested element belongs to container given by iterators.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Info the class of object info.
	 *  \tparam T the type of compared field.
	 *  \tparam Iter the iterator type.
	 *  \param wsk pointer to tested member.
	 *  \param b the iterator to the first element of container.
	 *  \param e the iterator to the past-the-last element of container.
	 *  \related FieldContainerChooser
	 *  \ingroup DMchooser*/
	template< class Info, class T, class Iter > FieldContainerChooser< Info,T,Iter >
		fieldChoose( T Info::*wsk, Iter b, Iter e ) { return FieldContainerChooser< Info,T,Iter >( wsk,b,e ); }

	/** \brief Is key in associative container chooser.
	 *
	 *  Function object that checks if the element (pointer) is  a key in associative container defined in constructor.
	 *  Mind that associative container is copied, which is a waste of resources, however result is not influenced by further changes in container.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocHasChooser
	{
		Cont cont;/**<\brief The container with approved elements.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocHasChooser< Cont > ChoosersSelfType;

		/**\brief Constructor
		 *
		 * The constructor sets the associative container.*/
		AssocHasChooser( const Cont &arg = Cont() ): cont( arg ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true if and only if \a elem is a key in the container.
		 *  \param elem the checked object.
		 *  \param graph the considered graph.
		 *  \return true if and only if \a elem is a key in the container. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph & ) const { return cont.hasKey( elem ); }
	};

	/** \brief Generating  function of AssocHasChooser.
	 *
	 *  AssocHasChooser function object is generated. The functor test if element is a key in associative container.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \param arg the associative container with approved elements.
	 *  \related AssocHasChooser
	 *  \ingroup DMchooser*/
	template< class Cont >
		AssocHasChooser< Cont > assocKeyChoose( Cont arg ) { return AssocHasChooser< Cont >( arg ); }

	/** \brief Has true mapped value chooser.
	 *
	 *  Function object that checks if the element has a mapped value convertible to true in the associative container defined in constructor.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocBoolChooser
	{
		Cont cont;/**<\brief The container with approved elements.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocBoolChooser< Cont > ChoosersSelfType;

		/**\brief Constructor
		 *
		 * The constructor sets the associative container.*/
		AssocBoolChooser(const Cont &arg = Cont()) : cont(arg) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true if and only if \a elem is a key and the mapped value is convertible to true.
		 *  \param elem the checked object.
		 *  \param graph the considered graph.
		 *  \return true if and only if \a elem is a key in the container and the mapped value is true (or convertible to true). */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph & ) const { return cont.hasKey( elem ) && (bool)cont[elem]; }
	};

	/** \brief Generating  function of AssocBoolChooser.
	 *
	 *  AssocBoolChooser function object is generated. The functor test if element is a key in associative container
	 *  and if the mapped value is convertible to true.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \param arg the associative container that maps elements to values convertible to Boolean type.
	 *  \related AssocBoolChooser
	 *  \ingroup DMchooser*/
	template< class Cont >
		AssocBoolChooser< Cont > assocChoose( Cont arg ) { return AssocBoolChooser< Cont >( arg ); }

	/** \brief Mapped value chooser.
	 *
	 *  The chooser is equipped with associative container. Each call of function call operator test if element mapped value matches the given value.
	 *  Furthermore, the element needs to be a key in the container. Both, the associative container and the value to compare are set up in constructor.
	 *  The associative container is copied. Hence, further change in container do not influence the effect.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocValChooser
	{
		Cont cont;/**<\brief The container with approved elements.*/

		typename Cont::ValType val;/**<\brief The desired value of mapped value.*/
		typedef typename Cont::ValType SelfValType;
		/** \brief Chooser obligatory type.
		 *
		 *  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocValChooser< Cont > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The constructor sets up the associative container and chosen mapped value.*/
		AssocValChooser( const Cont &arg = Cont(), typename Cont::ValType aval = SelfValType() ):
			cont( arg ),val( aval ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and mapped value matches \a val.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value equals \a val. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph & ) const { return cont.hasKey( elem ) && cont[elem] == val; }
	};

	/** \brief Generating  function of AssocValChooser.
	 *
	 *  AssocValChooser function object is generated. The functor test if element is a key in associative container
	 *  and if the mapped value equals \a val.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \param arg the associative container.
	 *  \related AssocValChooser
	 *  \ingroup DMchooser*/
	template< class Cont > AssocValChooser< Cont >
		assocChoose( Cont arg, typename Cont::ValType val ) { return AssocValChooser< Cont >( arg,val ); }

	/** \brief Choose elements for which mapped value less then common value.
	 *
	 *  The functor is equipped with associative container. Each call of function call operator tests if element mapped value is smaller then prespcified value.
	 *  Furthermore, the element needs to be a key in the container. Both, the associative container and the value to compare are set up in constructor.
	 *  The chooser works for both edges and vertices. Mind that mapped values in container must allow operator<.
	 *  The associative container is copied. Hence, further change in container do not influence the effect.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocValChooserL
	{
		Cont cont;/**<\brief The associative container for elements.*/

		typename Cont::ValType val;/**<\brief The desired value of mapped value.*/
		typedef typename Cont::ValType SelfValType;

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocValChooserL< Cont > ChoosersSelfType;

		/** \brief Constructor
		*
		*  The constructor sets up the associative container and the value to compare.*/
		AssocValChooserL(const Cont &arg = Cont(), typename Cont::ValType aval = SelfValType()) :
			cont( arg ),val( aval ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and mapped value is lower then \a val.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value is smaller then \a val. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph & ) const { return cont.hasKey( elem ) && cont[elem] < val; }
	};

	/** \brief Generating  function of AssocValChooserL.
	*
	*  AssocValChooserL function object is generated. The functor test if element is a key in associative container
	*  and if the mapped value is lower then \a val.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Cont the type of associative container.
	*  \param arg the associative container.
	*  \related AssocValChooserL
	*  \ingroup DMchooser*/
	template< class Cont > AssocValChooserL< Cont >
		assocChooseL( Cont arg, typename Cont::ValType val ) { return AssocValChooserL< Cont >( arg,val ); }

	/** \brief Choose elements for which mapped value greater then common value.
	*
	*  The functor is equipped with associative container. Each call of function call operator tests if element mapped value is greater then prespcified value.
	*  Furthermore, the element needs to be a key in the container. Both, the associative container and the value to compare are set up in constructor.
	*  The chooser works for both edges and vertices. Mind that mapped values in container must allow operator>.
	*  The associative container is copied. Hence, further change in container do not influence the effect.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Cont the type of associative container.
	*  \ingroup DMchooser */
	template< class Cont > struct AssocValChooserG
	{
		Cont cont;/**<\brief The associative container with elements and tested mapped values.*/

		typename Cont::ValType val;/**<\brief The desired value of mapped value.*/
		typedef typename Cont::ValType SelfValType;

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocValChooserG< Cont > ChoosersSelfType;

		/** \brief Constructor
		*
		*  The constructor sets up the associative container and the value to compare.*/
		AssocValChooserG(const Cont &arg = Cont(), typename Cont::ValType aval = SelfValType()) :
			cont( arg ),val( aval ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and the mapped value is greater then \a val.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value is greater then \a val. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph & ) const { return cont.hasKey( elem ) && cont[elem] > val; }
	};

	/** \brief Generating  function of AssocValChooserG.
	*
	*  AssocValChooserG function object is generated. The functor test if element is a key in associative container
	*  and if the mapped value is greater then \a val.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Cont the type of associative container.
	*  \param arg the associative container.
	*  \related AssocValChooserG
	*  \ingroup DMchooser*/
	template< class Cont > AssocValChooserG< Cont >
		assocChooseG( Cont arg, typename Cont::ValType val ) { return AssocValChooserG< Cont >( arg,val ); }

	/** \brief Choose if mapped value belongs to set.
	 *
	 *  Function object that checks if the element mapped value belongs to the prespecified set.
	 *  Furthermore, the element needs to be a key in the associative container.
	 *  Both, the associative container and the set are defined in the constructor.
	 *  They are copied. Hence, their further change do not influence the effect.
	 *  The chooser works with both edges and vertices.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocSetChooser
	{
		Cont cont;/**<\brief The associative container for elements.*/

		Koala::Set< typename Cont::ValType > set;/**<\brief The set with approved mapped values.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocSetChooser< Cont > ChoosersSelfType;

		/** \brief Constructor
		*
		*  The constructor sets up the associative container and the set of approved mapped values to compare.*/
		AssocSetChooser(const Cont &arg = Cont(), const Koala::Set< typename Cont::ValType > &aset =
			Koala::Set< typename Cont::ValType >() ): cont( arg ),set( aset ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and the mapped value belongs to set \a set.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value belongs to set \a set. */
		template< class Elem, class Graph > bool
			operator()( Elem *elem, const Graph & ) const { return cont.hasKey( elem ) && set.isElement( cont[elem] ); }
	};

	/** \brief Generating  function of AssocSetChooser.
	*
	*  AssocSetChooser function object is generated. The functor test if element is a key in associative container
	*  and if the mapped value belongs to the set.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Cont the type of associative container.
	*  \param arg the associative container.
	*  \param set the set with approved mapped values.
	*  \related AssocSetChooser
	*  \ingroup DMchooser*/
	template< class Cont > AssocSetChooser< Cont > assocChoose(Cont arg,
		const Koala::Set< typename Cont::ValType > &set ) { return AssocSetChooser< Cont >( arg,set ); }

	/** \brief Choose if mapped value belongs to container.
	 *
	 *  Function object that checks if the element mapped value belongs to the prespecified another container
	 *  (given with iterators \wikipath{iterator, Get more data about iterators).
	 *  Furthermore, the element needs to be a key in the associative container.
	 *  Both, the associative container and the container are defined in the constructor.
	 *  The associative array is copied. But the container with approved mapped values is not and it is user prerogative to keep it valid.
	 *  The container should be stl-like and must allow stl::find algorithm.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \tparma Iter the iterator type of container to search mapped values in for.
	 *  \ingroup DMchooser */
	template< class Cont, class Iter > struct AssocContainerChooser
	{
		Iter begin/**<\brief the iterator to the first element of the container.*/, end/**<\brief the iterator to the past-the-end element of the container.*/;
		Cont cont;/**<\brief The associative container*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocContainerChooser< Cont, Iter > ChoosersSelfType;

		/**\brief Constructor
		 *
		 *  In the constructor the associative container is copied and the container with approved mapped values is set up (via iterators.
		 *  \param acont the associative container.
		 *  \param abegin the iterator to the first element of container with approved mapped values.
		 *  \param aend the iterator to the past-the-last element of container with approved mapped values.*/
		AssocContainerChooser( const Cont &acont = Cont(), Iter abegin = Iter(), Iter aend = Iter() ):
			cont( acont ), begin( abegin ), end( aend ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true if and only if \a elem is a key in associative array and the mapped value belongs container given by iterators.
		 *  \param elem the checked object.
		 *  \param graph the considered graph.
		 *  \return true if and only if \a elem is a key in the container and the mapped value belongs to the container with approved mapped values. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &) const
			{ return cont.hasKey( elem ) && std::find( begin,end,cont[elem] ) != end; }
	};

	/** \brief Generating  function of AssocContainerChooser.
	 *
	 *  AssocContainerChooser function object is generated. The functor test if element is a key in associative container
	 *  and if the mapped value is an element in contaier given by iterators \a begin \a end.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \tparam Iter the iterator type.
	 *  \param cont the associative container.
	 *  \param begin the iterator to the first element of container with approved mapped values.
	 *  \param end the iterator to the past-the-last element of container with approved mapped values.
	 *  \related AssocContainerChooser
	 *  \ingroup DMchooser*/
	template< class Cont, class Iter > AssocContainerChooser< Cont, Iter >
		assocChoose( Cont cont, Iter begin, Iter end ) { return AssocContainerChooser< Cont,Iter >( cont,begin,end ); }

	/** \brief Choose if functor returns true for mapped value.
	 *
	 *  The function object is equipped with functor and associative container both set up in constructor.
	 *  The element to be chosen must be a key in associative container and function should return value convertible to true for mapped  value.
	 *  The chooser works for both edges and vertices.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \tparma Obj the functor class.
	 *  \ingroup DMchooser */
	template< class Cont, class Obj > struct AssocObjChooser
	{
		mutable Obj functor;/**<\brief The function object.*/
		Cont cont;/**<\brief The associative container.*/

		/** \brief Chooser obligatory type.
		 *
		 *  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocObjChooser< Cont, Obj > ChoosersSelfType;
		/**\brief Constructor
		 *
		 * The associative container and testing function object are set up. */
		AssocObjChooser( const Cont &acont = Cont(), Obj arg = Obj() ): cont( acont ), functor( arg ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning true if and only if \a elem is a key in associative array and the function object returns true for mapped value.
		 *  \param elem the checked object.
		 *  \param graph the considered graph.
		 *  \return true if and only if \a elem is a key in the container and the functor returns true for mapped value. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &graph) const
			{ return cont.hasKey( elem ) && (bool)functor( cont[elem] ); }
	};

	/** \brief Generating  function of AssocObjChooser.
	*
	*  AssocObjChooser function object is generated. The functor test if the element is a key in associative container
	*  and if functor \a arg returns valeu convertible to true for mapped value.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Cont the type of associative container.
	*  \tparam Obj the function object type.
	*  \param cont the associative container.
	*  \param arg the function object
	*  \related AssocObjChooser
	*  \ingroup DMchooser*/
	template< class Cont, class Obj > AssocObjChooser< Cont, Obj >
		assocFChoose( Cont cont, Obj arg ) { return AssocObjChooser< Cont,Obj >( cont,arg ); }

	/** \brief Is key in associative container chooser.
	 *
	 *  Function object that checks if the element (pointer) is  a key in associative container defined in constructor.
	 *  This is specialization of AssocHasChooser for pointers and that associative container is not copied.
	 *  It is users prerogative to keep the container up to date.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocHasChooser< Cont * >
	{
		const Cont *cont;/**<\brief The pointer to the associative container.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocHasChooser< Cont * > ChoosersSelfType;

		/**\brief Container
		 *
		 *  The constructor sets the associative container.*/
		AssocHasChooser(const Cont *arg = 0) : cont(arg) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in the container.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container. */
		template< class Elem, class Graph >
			bool operator()( Elem *elem, const Graph &) const { return cont->hasKey( elem ); }
	};

	/** \brief Generating  function of AssocHasChooser< Cont * >.
	 *
	 *  AssocHasChooser< Cont * > function object is generated. The functor test if element is a key in associative container.
	 *  The chooser is a specialized for pointers version of AssocHasChooser which do not copy the associative container.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \param arg the pointer to associative container with approved elements.
	 *  \related AssocHasChooser
	 *  \ingroup DMchooser*/
	template< class Cont >
		AssocHasChooser< Cont * > extAssocKeyChoose( const Cont *arg ) { return AssocHasChooser< Cont * >( arg ); }

	/** \brief Has true mapped value chooser.
	 *
	 *  Function object that checks if the element has a mapped value convertible to true in the associative container defined in constructor.
	 *  The chooser works for both edges and vertices.
	 *  This is specialization of AssocBoolChooser for pointers and that associative container is not copied.
	 *  It is users prerogative to keep the container up to date.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocBoolChooser< Cont * >
	{
		const Cont *cont;/**<\brief The pointer to the associative container.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocBoolChooser< Cont * > ChoosersSelfType;

		/**\brief Constructor
		*
		* The constructor sets the associative container.*/
		AssocBoolChooser(const Cont *arg = 0) : cont(arg) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key and the mapped value is convertible to true.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value is true (or convertible to true). */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &) const
			{ return cont->hasKey( elem ) && (bool)cont->operator[]( elem ); }
	};

	/** \brief Generating  function of AssocBoolChooser< Cont * >.
	 *
	 *  AssocBoolChooser< Cont * > function object is generated. The functor test if element is a key in associative container
	 *  and if the mapped value is convertible to true. The chooser is specialized version of AssocBoolChooser for pointers
	 *  which do not copy the container.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \param arg the pointer to associative container that maps elements to values convertible to Boolean type.
	 *  \related AssocBoolChooser
	 *  \ingroup DMchooser*/
	template< class Cont >
		AssocBoolChooser< Cont * > extAssocChoose( const Cont *arg ) { return AssocBoolChooser< Cont * >( arg ); }

	/** \brief Mapped value chooser.
	 *
	 *  The chooser is equipped with associative container. Each call of function call operator test if element mapped value matches the given value.
	 *  Furthermore, the element needs to be a key in the container. Both, the associative container and the value to compare are set up in constructor.
	 *  This is specialization of AssocHasChooser for pointers and that associative container is not copied.
	 *  It is users prerogative to keep the container up to date.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocValChooser< Cont * >
	{
		const Cont *cont;/**<\brief The pointer to the associative container.*/

		typename Cont::ValType val;/**<\brief The desired value of mapped value.*/
		typedef typename Cont::ValType SelfValType;

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocValChooser< Cont * > ChoosersSelfType;

		/** \brief Constructor
		*
		*  The constructor sets up the associative container and chosen mapped value.*/
		AssocValChooser(const Cont *arg = 0, typename Cont::ValType aval = SelfValType()) :
			cont( arg ), val( aval ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and mapped value matches \a val.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value equals \a val. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &) const
			{ return cont->hasKey( elem ) && cont->operator[]( elem ) == val; }
	};

	/** \brief Generating  function of AssocValChooser< Cont * >.
	 *
	 *  AssocValChooser< Cont * > function object is generated. The functor test if element is a key in associative container
	 *  and if the mapped value equals \a val. This a specialization of AssocValChooser for pointers
	 *  and the associative container is not copied.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \param arg the pointer to associative container.
	 *  \related AssocValChooser
	 *  \ingroup DMchooser*/
	template< class Cont > AssocValChooser< Cont * >
		extAssocChoose( const Cont *arg, typename Cont::ValType val ) { return AssocValChooser< Cont * >( arg,val ); }

	/** \brief Choose elements for which mapped value greater then common value.
	 *
	 *  The functor is equipped with associative container. Each call of function call operator tests if element mapped value is greater then prespcified value.
	 *  Furthermore, the element needs to be a key in the container. Both, the associative container and the value to compare are set up in constructor.
	 *  The chooser works for both edges and vertices. Mind that mapped values in container must allow operator>.
	 *  This is specialization of AssocValChooserG for pointers and that associative container is not copied.
	 *  It is users prerogative to keep the container up to date.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocValChooserG< Cont * >
	{
		const Cont *cont;/**<\brief The pointer to the associative container.*/

		typename Cont::ValType val;/**<\brief The desired value of mapped value.*/
		typedef typename Cont::ValType SelfValType;

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocValChooserG< Cont * > ChoosersSelfType;

		/** \brief Constructor
		*
		*  The constructor sets up the associative container and the value to compare.*/
		AssocValChooserG(const Cont *arg = 0, typename Cont::ValType aval = SelfValType()) :
			cont( arg ), val( aval ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and the mapped value is greater then \a val.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value is greater then \a val. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &) const
			{ return cont->hasKey( elem ) && cont->operator[]( elem ) > val; }
	};

	/** \brief Generating  function of AssocValChooserG< Cont * >.
	*
	*  AssocValChooserG< Cont * > function object is generated. The functor test if element is a key in associative container
	*  and if the mapped value is greater then \a val. This is a specialized version of AssocValChooserG for pointers which
	*  do not copy the associative container.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Cont the type of associative container.
	*  \param arg the pointer to associative container.
	*  \related AssocValChooserG
	*  \ingroup DMchooser*/
	template< class Cont > AssocValChooserG< Cont * >
		extAssocChooseG( const Cont *arg, typename Cont::ValType val ) { return AssocValChooserG< Cont * >( arg,val ); }

	/** \brief Choose elements for which mapped value less then common value.
	 *
	 *  The functor is equipped with pointer to associative container. Each call of function call operator tests if element mapped value is smaller then prespcified value.
	 *  Furthermore, the element needs to be a key in the container. Both, the associative container and the value to compare are set up in constructor.
	 *  The chooser works for both edges and vertices. Mind that mapped values in container must allow operator<.
	 *  This is specialization of AssocValChooserL for pointers and that associative container is not copied.
	 *  It is users prerogative to keep the container up to date.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocValChooserL< Cont * >
	{
		const Cont *cont;/**<\brief The pointer to the associative container.*/

		typename Cont::ValType val;/**<\brief The desired value of mapped value.*/
		typedef typename Cont::ValType SelfValType;

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocValChooserL< Cont * > ChoosersSelfType;

		/** \brief Constructor
		*
		*  The constructor sets up the associative container and the value to compare.*/
		AssocValChooserL(const Cont *arg = 0, typename Cont::ValType aval = SelfValType()) :
			cont( arg ), val( aval ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and mapped value is lower then \a val.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value is smaller then \a val. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &) const
			{ return cont->hasKey( elem ) && cont->operator[]( elem ) < val; }
	};

		/** \brief Generating  function of AssocValChooserL< Cont * >.
		*
		*  AssocValChooserL< Cont * > function object is generated. The functor test if element is a key in associative container
		*  and if the mapped value is lower then \a val. The chooser is a specialized version of AssocValChooserL
		*  for pointers which do not copy the associative container.
		*  \wikipath{chooser, Get more information about choosers.}
		*  \tparam Cont the type of associative container.
		*  \param arg the pointer to associative container.
		*  \related AssocValChooserL
		*  \ingroup DMchooser*/
		template< class Cont > AssocValChooserL< Cont * >
		extAssocChooseL( const Cont *arg, typename Cont::ValType val ) { return AssocValChooserL< Cont * >( arg,val ); }

	/** \brief Choose if mapped value belongs to set.
	 *
	 *  Function object that checks if the element mapped value belongs to the prespecified set.
	 *  Furthermore, the element needs to be a key in the associative container.
	 *  Both, the associative container and the set are defined in the constructor.
	 *  This is specialization of AssocSetChooser for pointers and that associative container is not copied.
	 *  It is users prerogative to keep the container up to date. However, that the set is copied.
	 *  The chooser works with both edges and vertices.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \ingroup DMchooser */
	template< class Cont > struct AssocSetChooser< Cont * >
	{
		const Cont *cont;/**<\brief The pointer to the associative container.*/

		Koala::Set< typename Cont::ValType > set;/**<\brief The set with approved mapped values.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocSetChooser< Cont * > ChoosersSelfType;

		/** \brief Constructor
		*
		*  The constructor sets up the associative container and the set of approved mapped values to compare.*/
		AssocSetChooser(const Cont *arg = 0, const Koala::Set< typename Cont::ValType > &aset =
			Koala::Set< typename Cont::ValType>() ): cont( arg ), set( aset ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and the mapped value belongs to set \a set.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value belongs to set \a set. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &) const
			{ return cont->hasKey( elem ) && set.isElement( cont->operator[]( elem ) ); }
	};

	/** \brief Generating  function of AssocSetChooser< Cont * >.
	*
	*  AssocSetChooser< Cont * > function object is generated. The functor test if element is a key in associative container
	*  and if the mapped value belongs to the set. The chooser is a specialized version of AssocSetChooser for which only
	*  the set is copied and the associative container is delivered by pointer.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Cont the type of associative container.
	*  \param arg the pointer to associative container.
	*  \param set the set with approved mapped values.
	*  \related AssocSetChooser
	*  \ingroup DMchooser*/
	template< class Cont > AssocSetChooser< Cont * > extAssocChoose( const Cont *arg,
		const Koala::Set< typename Cont::ValType > &set ) { return AssocSetChooser< Cont * >( arg,set ); }

	/** \brief Choose if mapped value belongs to container.
	*
	*  Function object that checks if the element mapped value belongs to the prespecified another container
	*  (given with iterators \wikipath{iterator, Get more data about iterators).
	*  Furthermore, the element needs to be a key in the associative container.
	*  Both, the associative container and the container are defined in the constructor.
	*  The container with approved mapped values is not copied.
	*  Since, this is a specialization of AssocContainerChooser for pointers also that associative container is not copied.
	*  It is users prerogative to keep the containers up to date.
 	*  The container with approved values should be stl-like and must allow stl::find algorithm.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Cont the type of associative container.
	*  \tparma Iter the iterator type of container to search mapped values in for.
	*  \ingroup DMchooser */
	template< class Cont, class Iter > struct AssocContainerChooser< Cont *, Iter >
	{
		Iter begin/**<\brief the iterator to the first element of the container.*/, end/**<\brief the iterator to the past-the-end element of the container.*/;
		const Cont *cont;/**<\brief The pointer to the associative container.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocContainerChooser< Cont *, Iter > ChoosersSelfType;

		/**\brief Constructor
		*
		*  In the constructor the associative container is copied and the container with approved mapped values is set up (via iterators.
		*  \param acont the associative container.
		*  \param abegin the iterator to the first element of container with approved mapped values.
		*  \param aend the iterator to the past-the-last element of container with approved mapped values.*/
		AssocContainerChooser(const Cont *acont = 0, Iter abegin = Iter(), Iter aend = Iter()) :
			cont( acont ), begin( abegin ), end( aend ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and the mapped value belongs container given by iterators.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the mapped value belongs to the container with approved mapped values. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &) const;
	};

	/** \brief Generating  function of AssocContainerChooser< Cont *,Iter >.
	 *
	 *  AssocContainerChooser< Cont *,Iter > function object is generated. The functor test if element is a key in associative container
	 *  and if the mapped value is an element in contaier given by iterators \a begin \a end.
	 *  In this specialization of AssocContainerChooser the associative container \a cont is not copied.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \tparam Iter the iterator type.
	 *  \param cont the pointer associative container.
	 *  \param begin the iterator to the first element of container with approved mapped values.
	 *  \param end the iterator to the past-the-last element of container with approved mapped values.
	 *  \related AssocContainerChooser
	 *  \ingroup DMchooser*/
	template< class Cont, class Iter > AssocContainerChooser< Cont *,Iter >
		extAssocChoose(const Cont *cont, Iter begin, Iter end )
	{ return AssocContainerChooser< Cont *,Iter >( cont,begin,end ); }

	/** \brief Choose if functor returns true for mapped value.
	 *
	 *  The function object is equipped with functor and associative container both set up in constructor.
	 *  The element to be chosen must be a key in associative container and function should return value convertible to true for mapped  value.
	 *  The chooser works for both edges and vertices.
	 *  This is specialization of AssocObjChooser for pointers and that associative container is not copied.
	 *  It is users prerogative to keep the container up to date.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Cont the type of associative container.
	 *  \tparma Obj the functor class.
	 *  \ingroup DMchooser */
	template< class Cont, class Obj > struct AssocObjChooser< Cont *,Obj >
	{
		mutable Obj functor;/**<\brief The function object.*/
		const Cont *cont;/**<\brief The pointer to the associative container.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AssocObjChooser< Cont *, Obj > ChoosersSelfType;

		/**\brief Constructor
		*
		* The associative container and testing function object are set up. */
		AssocObjChooser(const Cont *acont = 0, Obj arg = Obj()) : cont(acont), functor(arg) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if \a elem is a key in associative array and the function object returns true for mapped value.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if \a elem is a key in the container and the functor returns true for mapped value. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &graph) const
			{ return cont->hasKey( elem ) && (bool)functor( cont->operator[]( elem ) ); }
	};

	/** \brief Generating  function of AssocObjChooser< Cont *,Obj >.
	*
	*  AssocObjChooser< Cont *,Obj > function object is generated. The functor test if the element is a key in associative container
	*  and if functor \a arg returns valeu convertible to true for mapped value. The chooser is a specialized version of AssocObjChooser
	*  which do not copy the associative container.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Cont the type of associative container.
	*  \tparam Obj the function object type.
	*  \param cont the pointer to associative container.
	*  \param arg the function object
	*  \related AssocObjChooser
	*  \ingroup DMchooser*/
	template< class Cont, class Obj > AssocObjChooser< Cont *,Obj >
		extAssocFChoose( const Cont *cont, Obj arg ) { return AssocObjChooser< Cont *,Obj >( cont,arg ); }

	/** \brief Or chooser.
	 *
	 *  The function object that joins two choosers. It returns true value if and only if the first one or the second one return true.
	 *  The chooser works for both edges and vertices. \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Ch1 the first chooser class.
	 *  \tparma Ch2 the second chooser class.
	 *  \ingroup DMchooser */
	template< class Ch1, class Ch2 > struct OrChooser
	{
		Ch1 ch1;/**<\brief The first chooser. */
		Ch2 ch2;/**<\brief The second chooser. */

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef OrChooser< Ch1, Ch2 > ChoosersSelfType;
        /**\brief Constructor
		 *
		 * The constructor sets up the choosers. */
		OrChooser( Ch1 a = Ch1(), Ch2 b = Ch2() ): ch1( a ), ch2( b ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if chooser \a ch1 or \a ch2 returns true for given element \a elem.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if choosers \a ch1 or \a ch2 return true. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &graph) const
			{ return (ch1( elem,graph ) || ch2( elem,graph )); }
	};

	/** \brief Generating  function of OrChooser.
	*
	*  OrChooser function object is generated. The functor chooses elements that are chosen for at least one of choosers \a ch1 or \a ch2.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Ch1 the type of the first chooser.
	*  \tparam Ch2 the type of the second chooser.
	*  \param a the first chooser.
	*  \param b the second chooser
	*  \related OrChooser
	*  \ingroup DMchooser*/
	template< class  Ch1, class Ch2 > OrChooser< Ch1, Ch2 >
		orChoose( Ch1 a, Ch2 b ) { return OrChooser< Ch1,Ch2 >( a,b ); }

	/** \brief The overloaded operator||. Chooser alternative.
	 *
	 *  The operator calls the generating  function of OrChooser i.e. generate chooser that joins two choosers with logic or operator.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \related OrChooser
	 *  \ingroup DMchooser*/
	template <class  Ch1, class Ch2> OrChooser< typename Ch1::ChoosersSelfType,typename Ch2::ChoosersSelfType >
		operator||( Ch1 a, Ch2 b ) { return OrChooser< Ch1,Ch2 >( a,b ); }

	/** \brief And chooser.
	 *
	 *  The function object that joins two choosers. It returns true value if and only if both choosers return true when called with the element.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Ch1 the first chooser class.
	 *  \tparma Ch2 the second chooser class.
	 *  \ingroup DMchooser */
	template< class Ch1, class Ch2 > struct AndChooser
	{
		Ch1 ch1;/**<\brief The first chooser. */
		Ch2 ch2;/**<\brief The second chooser. */

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef AndChooser< Ch1, Ch2 > ChoosersSelfType;

		/**\brief Constructor
		*
		* The constructor sets up the choosers. */
		AndChooser(Ch1 a = Ch1(), Ch2 b = Ch2()) : ch1(a), ch2(b) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if both choosers \a ch1 and \a ch2 return true for given element \a elem.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if choosers \a ch1 and \a ch2 return true. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &graph) const
			{ return (ch1( elem,graph ) && ch2( elem,graph )); }
	};

	/** \brief Generating  function of AndChooser.
	*
	*  AndChooser function object is generated. The functor chooses elements that are chosen for both choosers \a ch1 and \a ch2.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Ch1 the type of the first chooser.
	*  \tparam Ch2 the type of the second chooser.
	*  \param a the first chooser.
	*  \param b the second chooser
	*  \related AndChooser
	*  \ingroup DMchooser*/
	template< class  Ch1, class Ch2 > AndChooser< Ch1, Ch2 >
		andChoose( Ch1 a, Ch2 b ) { return AndChooser< Ch1,Ch2 >( a,b ); }

	/** \brief The overloaded operator&&. Chooser conjunction.
	 *
	 *  The operator calls the generating  function of AndChooser i.e. it generates chooser that joins two choosers with logic and operator.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \related AndChooser
	 *  \ingroup DMchooser*/
	template< class Ch1, class Ch2 > AndChooser< typename Ch1::ChoosersSelfType,typename Ch2::ChoosersSelfType >
		operator&&( Ch1 a, Ch2 b ) { return AndChooser< Ch1,Ch2 >( a,b ); }

	/** \brief Xor chooser.
	 *
	 *  The function object that joins two choosers. It returns true value if and only if either the first one or the second one return true
	 *  (both may not be true).
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Ch1 the first chooser class.
	 *  \tparma Ch2 the second chooser class.
	 *  \ingroup DMchooser */
	template< class Ch1, class Ch2 > struct XorChooser
	{
		Ch1 ch1;/**<\brief The first chooser. */
		Ch2 ch2;/**<\brief The second chooser. */

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef XorChooser< Ch1, Ch2 > ChoosersSelfType;

		/**\brief Constructor
		*
		* The constructor sets up the choosers. */
		XorChooser(Ch1 a = Ch1(), Ch2 b = Ch2()) : ch1(a), ch2(b) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only if exactly one chooser either \a ch1 or \a ch2 return true for given element \a elem.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return true if and only if only one of choosers \a ch1 and \a ch2 return true. */
		template< class Elem, class Graph > bool operator()(Elem *elem, const Graph &graph) const
			{ return (ch1( elem,graph ) != ch2( elem,graph )); }
	};

	/** \brief Generating  function of XorChooser.
	*
	*  XorChooser function object is generated. The functor chooses elements for which exactly one of choosers \a ch1 and \a ch2 returns true.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Ch1 the type of the first chooser.
	*  \tparam Ch2 the type of the second chooser.
	*  \param a the first chooser.
	*  \param b the second chooser
	*  \related XorChooser
	*  \ingroup DMchooser*/
	template< class Ch1, class Ch2 > XorChooser< Ch1, Ch2 >
		xorChoose( Ch1 a, Ch2 b ) { return XorChooser< Ch1,Ch2 >( a,b ); }

	/** \brief The overloaded operator^. Chooser exclusive or.
	 *
	 *  The operator calls the generating  function of XorChooser which generates chooser that joins two choosers with logic exclusive or operation.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Ch1 the type of the first chooser.
	 *  \tparam Ch2 the type of the second chooser.
	 *  \param a the first chooser.
	 *  \param b the second chooser
	 *  \related XorChooser
	 *  \ingroup DMchooser*/
	template< class Ch1, class Ch2 > XorChooser< typename Ch1::ChoosersSelfType,typename Ch2::ChoosersSelfType >
		operator^( Ch1 a, Ch2 b ) { return XorChooser< Ch1,Ch2 >( a,b ); }

	/** \brief Not chooser.
	 *
	 *  The function object that gives the opposite result to the given chooser.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Ch1 the chooser class.
	 *  \ingroup DMchooser */
	template< class Ch1 > struct NotChooser
	{
		Ch1 ch1;/**<\brief The negated chooser type.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef NotChooser< Ch1 > ChoosersSelfType;

		/**\brief Constructor
		 *
		 * The constructor sets up the negated chooser.*/
		NotChooser( Ch1 a = Ch1() ): ch1( a ) { }

		/** \brief Overloaded operator()
		*
		*  Function call operator returning true if and only chooser \a ch1 returns false.
		*  \param elem the checked object.
		*  \param graph the considered graph.
		*  \return negation of chooser \a ch1. */
		template< class Elem, class Graph >  bool operator()(Elem *elem, const Graph &graph) const
			{ return !ch1( elem,graph ); }
	};

	/** \brief Generating  function of NotChooser.
	*
	*  NotChooser function object is generated. The functor chooses elements that are not chosen by \a ch1.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Ch1 the chooser type.
	*  \param a the chooser.
	*  \related NotChooser
	*  \ingroup DMchooser*/
	template< class  Ch1 >
		NotChooser< Ch1 > notChoose( Ch1 a ) { return NotChooser< Ch1 >( a ); }

	/** \brief The overloaded operator!. Chooser negation.
	 *
	 *  The operator calls the generating  function of NotChooser i.e. generate chooser that negates the given chooser.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Ch1 the chooser type.
	 *  \param a the chooser.
	 *  \related NotChooser
	 *  \ingroup DMchooser*/
	template< class  Ch1 > NotChooser< typename Ch1::ChoosersSelfType >
		operator!( Ch1 a ) { return NotChooser< Ch1 >( a ); }

	/** \brief Choose vertices of given degree.
	 *
	 *  The function object that checks if the vertex degree equals given common value.
	 *  The degree is calculated with respect to Koala::EdgeDirection (\wikipath{EdgeDirection, Read more about EdgeDirection} mask like in method ConstGraphMethods::deg.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	struct VertDegValChooser
	{
		int deg; /**< \brief the desired degree.*/
		Koala::EdgeDirection type; /**< \brief the considered edge direction.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef VertDegValChooser ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The expected degree and the edge direction to consider are defined here.
		 *  \param adeg the expected degree.
		 *  \param atype the mask of considered edge directions.*/
		VertDegValChooser( int adeg = 0, Koala::EdgeDirection atype = Koala::EdAll ): deg( adeg ), type( atype ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning boolean value true if the vertex \a v is of degree \a deg.
		 *  \param v the tested vertex.
		 *  \param g reference to considered graph.
		 *  \return the true if degree of the vertex matches \a adeg, false otherwise. */
		template< class Graph > bool operator()( typename Graph::PVertex v, const Graph &g ) const
			{ return g.deg( v,type ) == deg; }
	};

	/** \brief Generating  function of VertDegChoose.
	 *
	 *  \param adeg the defined degree.
	 *  \param atype type of direction used for degree calculation.
	 *  \return chooser of type VertDegValChooser, which chooses vertices of degree \a adag exclusively.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \related VertDegValChooser
	 *  \ingroup DMchooser*/
	inline VertDegValChooser vertDegChoose( int adeg, Koala::EdgeDirection atype = Koala::EdAll )
		{ return VertDegValChooser( adeg,atype ); }

	/** \brief Choose vertices of degree less then.
	 *
	 *  The function object that checks if the vertex degree is less then the prespecified value.
	 *  The degree is calculated with respect to Koala::EdgeDirection (\wikipath{EdgeDirection, Read more about EdgeDirection} mask like in method ConstGraphMethods::deg.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	struct VertDegValChooserL
	{
		int deg; /**< \brief the strict upper bond for degree*/
		Koala::EdgeDirection type;/**< \brief the considered edge direction.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef VertDegValChooserL ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The strict upper bond for degree and the edge direction to consider are defined here.
		 *  \param adeg the expected degree.
		 *  \param atype the mask of considered edge directions.*/
		VertDegValChooserL( int adeg = 0, Koala::EdgeDirection atype = Koala::EdAll ): deg( adeg ), type( atype ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning boolean value true if vertex \a v degree is smaller than \a deg.
		 *  \param v the tested vertex.
		 *  \param g reference to considered graph.
		 *  \return the true if degree of the vertex is smaller than \a deg, false otherwise. */
		template< class Graph > bool operator()( typename Graph::PVertex v, const Graph &g ) const
			{ return g.deg( v,type ) < deg; }
	};

	/** \brief Generating  function of VertDegValChooserL.
	*
	*  \param adeg the defined degree.
	*  \param atype type of direction used for degree calculation.
	*  \return chooser of type VertDegValChooserL, which chooses vertices of degree less than \a adag.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \related VertDegValChooserL
	*  \ingroup DMchooser*/
	inline VertDegValChooserL vertDegChooseL(int adeg, Koala::EdgeDirection atype = Koala::EdAll)
		{ return VertDegValChooserL( adeg,atype ); }

	/** \brief Choose vertices of degree greater then.
	 *
	 *  The function object that checks if the vertex degree is greater then the prespecified value.
	 *  The degree is calculated with respect to Koala::EdgeDirection (\wikipath{EdgeDirection, Read more about EdgeDirection} mask like in method ConstGraphMethods::deg.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	struct VertDegValChooserG
	{
		int deg;/**< \brief the strict lower bond degree*/
		Koala::EdgeDirection type;/**< \brief the considered edge direction.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef VertDegValChooserG ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The strict lover bond for degree and the edge direction to consider are defined here.
		 *  \param adeg the expected degree.
		 *  \param atype the mask of considered edge directions.*/
		VertDegValChooserG( int adeg = 0, Koala::EdgeDirection atype = Koala::EdAll ): deg( adeg ), type( atype ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning boolean value true if the vertex \a v degree is greater than \a deg.
		 *  \param v the tested vertex.
		 *  \param g reference to considered graph.
		 *  \return the true if degree of the vertex is greater than \a deg, false otherwise. */
		template< class Graph > bool operator()( typename Graph::PVertex v, const Graph &g ) const
			{ return g.deg( v,type ) > deg; }
	};

	/** \brief Generating  function of VertDegValChooserG.
	*
	*  \param adeg the defined degree.
	*  \param atype type of direction used for degree calculation.
	*  \return chooser of type VertDegValChooserL, which chooses vertices of degree greater than \a adag.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \related VertDegValChooserG
	*  \ingroup DMchooser*/
	inline VertDegValChooserG vertDegChooseG(int adeg, Koala::EdgeDirection atype = Koala::EdAll)
		{ return VertDegValChooserG( adeg,atype ); }

	/** \brief Choose vertices of degree from set.
	 *
	 *  The function object that checks if the vertex degree belongs the set prespecified in constructor.
	 *  The degree is calculated with respect to Koala::EdgeDirection (\wikipath{EdgeDirection, Read more about EdgeDirection}) mask like in method ConstGraphMethods::deg.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	template< class Int > struct VertDegSetChooser
	{
		Koala::Set< Int > set;/**< \brief the set with acceptable degrees.*/
		Koala::EdgeDirection type;/**< \brief the considered edge direction.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef VertDegSetChooser< Int > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The set with acceptable degrees and the edge direction to consider are defined here.
		 *  \param aset the set with degrees.
		 *  \param atype the mask of considered edge directions.*/
		VertDegSetChooser( const Koala::Set< Int > &aset = Koala::Set< Int >(),
			Koala::EdgeDirection atype = Koala::EdAll ): set( aset ), type( atype ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning Boolean value true if the vertex \a v degree belongs to \a set.
		 *  \param v the tested vertex.
		 *  \param g reference to considered graph.
		 *  \return the true if degree of the vertex belongs to \a set, false otherwise. */
		template< class Graph > bool operator()( typename Graph::PVertex v, const Graph &g ) const
			{ return set.isElement( g.deg( v,type ) ); }
	};

	/** \brief Generating  function of VertDegSetChooser.
	*
	*  \param adeg the defined degree.
	*  \param atype type of direction used for degree calculation.
	*  \return chooser of type VertDegSetChooser, which chooses vertices as long as their degree belongs to \a aset.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \related VertDegSetChooser
	*  \ingroup DMchooser*/
	template< class Int > VertDegSetChooser< Int > vertDegChoose(Koala::Set< Int > aset,
		Koala::EdgeDirection atype = Koala::EdAll ) { return VertDegSetChooser< Int >( aset,atype ); }

	/** \brief Choose vertices of degree from container.
	 *
	 *  The function object that checks if the vertex degree is an element of the container prespecified in constructor.
	 *  The degree is calculated with respect to Koala::EdgeDirection (\wikipath{EdgeDirection, Read more about EdgeDirection}) mask like in method ConstGraphMethods::deg.
	 *  The container should be STL-like and it must allow std::fiend algorithm. The container is not copied and it is users prerogative to keep the container valid
	 *  and up to date. \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	template< class Iter > struct VertDegContainerChooser
	{
		Iter begin/**<\brief iterator to the first element of container*/, end/**<\brief iterator to the past-the-end element of container*/;
		Koala::EdgeDirection typ;/**< \brief the considered edge direction.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef VertDegContainerChooser< Iter > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The container with acceptable degree and the edge direction to consider are defined here.
		 *  \param abeg the iterator to the first element of container.
		 *  \param aend the iterator to the past-the-end element of container.
		 *  \param atype the mask of considered edge directions.*/
		VertDegContainerChooser( Iter abeg = Iter(), Iter aend = Iter(), Koala::EdgeDirection atype = Koala::EdAll ):
			begin( abeg ), end( aend ), typ( atype ) { }

		/** \brief Overloaded operator()
		 *
		 *  Function call operator returning Boolean value true if the vertex \a v degree belongs to the container.
		 *  \param v the tested vertex.
		 *  \param g reference to considered graph.
		 *  \return the true if degree of the vertex belongs to the container, false otherwise. */
		template< class Graph > bool operator()( typename Graph::PVertex v, const Graph &g ) const
			{ return std::find( begin,end,g.deg( v,typ ) ) != end; }
	};

	/** \brief Generating  function of VertDegContainerChooser.
	*
	*  \wikipath{chooser, Get more information about choosers.}
	*  \param begin iterator to the first element of container.
	*  \param end iterator to the past-the-end element of container
	*  \param atype type of direction used for degree calculation.
	*  \return chooser of type VertDegContainerChooser, which chooses vertices as long as their degree is an element in container.
	*  \related VertDegContainerChooser
	*  \ingroup DMchooser*/
	template< class Iter > VertDegContainerChooser< Iter > vertDegChoose(Iter begin, Iter end,
		Koala::EdgeDirection atype = Koala::EdAll ) { return VertDegContainerChooser< Iter >( begin,end,atype ); }

	/** \brief Choose vertices of degree accepted by functor.
	 *
	 *  The function object that for a given vertex tests if the vertex degree satisfy the functor defined in the constructor.
	 *  The functor must return value convertible to bool for various degree values.
	 *  The degree is calculated with respect to Koala::EdgeDirection (\wikipath{EdgeDirection, Read more about EdgeDirection}) mask like in method ConstGraphMethods::deg.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	template< class Obj > struct VertDegFunctorChooser
	{
		mutable Obj functor;/**< \brief the object function qualifying degrees.*/
		Koala::EdgeDirection typ;/**< \brief the considered edge direction.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef VertDegFunctorChooser< Obj > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The functor and the edge direction to consider are defined here.
		 *  \param afun the object function qualifying degrees.
		 *  \param atype the mask of considered edge directions.*/
		VertDegFunctorChooser( Obj afun = Obj(), Koala::EdgeDirection atype = Koala::EdAll ):
			functor( afun ), typ( atype ) { }

		/** \brief Overloaded operator()
		 *
		 *  The function call operator that returns value returns by \ a funktor casted to bool .
		 *  \param v the tested vertex.
		 *  \param g reference to considered graph.
		 *  \return the value returned by \a funktor. */
		template< class Graph > bool operator()( typename Graph::PVertex v, const Graph &g ) const
			{ return (bool)functor( g.deg( v,typ ) ); }
	};

	/** \brief Generating  function of VertDegFunctorChooser.
	*
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Obj the type of functor
	*  \param afun the function object approving vertex degrees.
	*  \param atype the type of direction used for degree calculation.
	*  \return chooser of type VertDegFunctorChooser, which chooses vertices as long as functor \a afun returns true for tested vertex degree.
	*  \related VertDegFunctorChooser
	*  \ingroup DMchooser*/
	template< class Obj > VertDegFunctorChooser< Obj > vertDegFChoose(Obj afun,
		Koala::EdgeDirection atype = Koala::EdAll ) { return VertDegFunctorChooser< Obj >( afun,atype ); }

	/** \brief Choose edges of given type.
	 *
	 *  The function object chooses the edges of type congruent with the Koala::EdgeType mask defined in constructor.
	 *  \wikipath{EdgeType,See possible values of EdgeType.}
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	struct EdgeTypeChooser
	{
		Koala::EdgeDirection mask;/**< \brief the considered edge direction.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef EdgeTypeChooser ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The direction of edge to choose is defined here.
		 *  \param amask the mask of considered edge types. \wikipath{EdgeType} */
		EdgeTypeChooser( Koala::EdgeDirection amask = Koala::EdAll ): mask( amask )
			{ mask |= (mask & Directed) ? Directed : 0; }

		/** \brief Overloaded operator()
		 *
		 *  The function call operator returning true if \a e type is congruent with Koala::EdgeType \a mask. \wikipath{EdgeType, See wiki for EdgeType}.
		 *  \param e the tested edge.
		 *  \param g reference to considered graph.
		 *  \return true if \a e type is congruent with the attribute \a mask, false otherwise. */
		template< class Graph > bool operator()( typename Graph::PEdge e, const Graph &g ) const
			{ return bool( g.getType( e ) & mask ); }
	};

	/** \brief Generating  function of EdgeTypeChooser.
	 *
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \param mask the Koala::EdgeType mask that defines the types of edges to be taken. \wikipath{EdgeType, See wiki for EdgeType}.
	 *  \return EdgeTypeChooser function object that chooses only edges with type congruent with \a mask.
	 *  \related EdgeTypeChooser
	 *  \ingroup DMchooser*/
	inline EdgeTypeChooser edgeTypeChoose( Koala::EdgeDirection mask ) { return EdgeTypeChooser( mask ); }

	/** \brief Choose edges for which first end satisfy given chooser.
	 *
	 *  The function object chooses edges for which the first end satisfy a functor (ex. some vertex chooser)  defined in constructor.
	 *  \tparam Ch the type of vertex chooser. The class should implement function call operator for two parameters: PVertex and a reference to graph.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	template< class Ch > struct EdgeFirstEndChooser
	{
		Ch ch;/**< \brief the function object that checks the first end of edge.*/

		/** \brief Chooser obligatory type.
		 *
		 *  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef EdgeFirstEndChooser< Ch > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Vertex chooser \a ch  is defined here.
		 *  \param funktor the functor assigned to attribute \a ch.*/
		EdgeFirstEndChooser( Ch funktor = Ch() ): ch( funktor ) { }

		/** \brief Overloaded operator()
		 *
		 *  \param e the tested edge.
		 *  \param g reference to considered graph.
		 *  \return the same value as functor called on the first end of \a e and the graph \a g. */
		template< class Graph > bool operator()( typename Graph::PEdge e, const Graph &g ) const
			{ return ch( g.getEdgeEnd1( e ),g ); }
	};

	/** \brief Generating  function of EdgeFirstEndChooser.
	 *
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Ch the type of vertex chooser.
	 *  \param ch the vertex chooser.
	 *  \return EdgeFirstEndChooser function object that chooses edges for which the first vertex satisfy chooser \a ch.
	 *  \related EdgeFirstEndChooser
	 *  \ingroup DMchooser*/
	template< class Ch > EdgeFirstEndChooser< Ch >
		edgeFirstEndChoose( Ch ch ) { return EdgeFirstEndChooser< Ch >( ch ); }

	/** \brief Choose edges for which second end satisfy given chooser.
	 *
	 *  The function object chooses edges for which the second end satisfy a functor (ex. some vertex chooser)  defined in constructor.
	 *  \tparam Ch the type of vertex chooser. The class should implement function call operator for two parameters: PVertex and a reference to graph.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	template <class Ch> struct EdgeSecondEndChooser
	{
		Ch ch;/**< \brief the function object that checks the second end of edge.*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef EdgeSecondEndChooser< Ch > ChoosersSelfType;

		/** \brief Constructor
		*
		*  Vertex chooser \a ch  is defined here.
		*  \param funktor the functor assigned to attribute \a ch.*/
		EdgeSecondEndChooser(Ch funktor = Ch()) : ch(funktor) { }

		/** \brief Overloaded operator()
		 *
		 *  \param e the tested edge.
		 *  \param g reference to considered graph.
		 *  \return the same value as functor called on the second end of \a e and graph \a g. */
		template< class Graph > bool operator()( typename Graph::PEdge e, const Graph &g ) const
			{ return ch( g.getEdgeEnd2( e ),g ); }
	};

	/** \brief Generating  function of EdgeSecondEndChooser.
	*
	*  \wikipath{chooser, Get more information about choosers.}
	*  \tparam Ch the type of vertex chooser.
	*  \param ch the vertex chooser.
	*  \return EdgeSecondEndChooser function object that chooses edges for which the first vertex satisfy chooser \a ch.
	*  \related EdgeSecondEndChooser
	*  \ingroup DMchooser*/
	template< class Ch > EdgeSecondEndChooser< Ch >
		edgeSecondEndChoose( Ch ch ) { return EdgeSecondEndChooser< Ch >( ch ); }

	/** \brief Choose if none of edge ends satisfy functor.
	 *
	 *  The function object chooses the edges in with none of its ends satisfies a functor (ex. some vertex chooser)  defined in constructor.
	 *  The chooser works only with edges.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \tparam Ch The type of chooser for vertices.
	 *  \ingroup DMchooser */
	template< class Ch > struct Edge0EndChooser // dla krawedzi
	{
		Ch ch;/**< \brief the function object that tests vertices (defined in constructor).*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef Edge0EndChooser< Ch > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  Vertex chooser \a ch is defined here.
		 *  \param funktor the chooser assigned to attribute \a ch.*/
		Edge0EndChooser( Ch funktor = Ch() ): ch( funktor ) { }

		/** \brief Overloaded operator()
		 *
		 *  \param e the tested edge.
		 *  \param g reference to considered graph.
		 *  \return true if and only if none of edge \a e ends satisfies the functor \a ch. */
		template< class Graph > bool operator()( typename Graph::PEdge e, const Graph &g ) const
			{ return !ch( g.getEdgeEnd1( e ),g ) && !ch( g.getEdgeEnd2( e ),g ); }
	};

	/** \brief Generating  function of Edge0EndChooser.
	 *
	 *  The function generates function object that chooses only those edges for which none of ends satisfy functor.
	 *  \tparam Ch the type of vertec chooser.
	 *  \param vertex chooser.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \related Edge0EndChooser
	 *  \ingroup DMchooser*/
	template< class Ch > Edge0EndChooser< Ch > edge0EndChoose(Ch ch) { return Edge0EndChooser< Ch >(ch); }

	/** \brief Choose if one edge end satisfy functor.
	 *
	 *  The function object chooses the edges in with one of its ends satisfies a functor (ex. some vertex chooser)  defined in constructor.
	 *  The chooser works for edges only.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	template< class Ch > struct Edge1EndChooser
	{
		Ch ch;/**< \brief the function object that tests vertices (defined in constructor).*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef Edge1EndChooser< Ch > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The \a ch functor is defined here.
		 *  \param funktor the vertex chooser assigned to attribute \a ch.*/
		Edge1EndChooser( Ch funktor = Ch() ): ch( funktor ) { }

		/** \brief Overloaded operator()
		 *
		 *  \param e the tested edge.
		 *  \param g reference to considered graph.
		 *  \return true if and only if one of edge \a e ends satisfies the functor \a ch. */
		template< class Graph > bool operator()( typename Graph::PEdge e, const Graph &g ) const
			{ return ch( g.getEdgeEnd1( e ),g ) != ch( g.getEdgeEnd2( e ),g ); }
	};

	/** \brief Generating  function of Edge1EndChooser.
	*
	*  The function generates Edge1EndChooser function object that chooses only those edges for which exactly one of ends satisfy functor.
	*  \tparam Ch the type of vertec chooser.
	*  \param vertex chooser.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \related Edge1EndChooser
	*  \ingroup DMchooser*/
	template< class Ch > Edge1EndChooser< Ch > edge1EndChoose(Ch ch) { return Edge1EndChooser< Ch >(ch); }

	/** \brief Choose if both edge ends satisfy functor.
	 *
	 *  The function object chooses the edges in with both ends satisfy a functor (ex. some vertex chooser)  defined in constructor.
	 *  The chooser works for edges only.
	 *  \wikipath{chooser, Get more information about choosers.}
	 *  \ingroup DMchooser */
	template< class Ch > struct Edge2EndChooser
	{
		Ch ch;/**< \brief the function object that tests vertices (defined in constructor).*/

		/** \brief Chooser obligatory type.
		*
		*  The type is obligatory for choosers in Koala. Logic operations (&&, ||, !, ^)  work properly as long as it is defined. */
		typedef Edge2EndChooser< Ch > ChoosersSelfType;

		/** \brief Constructor
		 *
		 *  The \a ch functor is defined here.
		 *  \param funktor the vertex chooser assigned to attribute \a ch.*/
		Edge2EndChooser( Ch funktor = Ch() ): ch( funktor ) { }

		/** \brief Overloaded operator()
		 *
		 *  \param e the tested edge.
		 *  \param g reference to considered graph.
		 *  \return true if and only if both edge \a e ends satisfy the functor \a ch. */
		template< class Graph > bool operator()( typename Graph::PEdge e, const Graph &g ) const
			{ return ch( g.getEdgeEnd1( e ),g ) && ch( g.getEdgeEnd2( e ),g ); }
	};

	/** \brief Generating  function of Edge2EndChooser.
	*
	*  The function generates Edge2EndChooser function object that chooses only those edges for which both ends satisfy functor.
	*  \tparam Ch the type of vertec chooser.
	*  \param vertex chooser.
	*  \wikipath{chooser, Get more information about choosers.}
	*  \related Edge2EndChooser
	*  \ingroup DMchooser*/
	template< class Ch > Edge2EndChooser< Ch > edge2EndChoose(Ch ch) { return Edge2EndChooser< Ch >(ch); }

	// Casters are functors that set values for info fields in new vertices/edges created during e.g. copying of graphs.
	// The values that are used base on original values.

    namespace Privates {

        namespace KonwTestSpace {


            template <class Dest,class Source> struct Przec {

                struct Lapacz {
                    int a;
    //                template <class T>
    //                Lapacz(T) : a(1) {}
                        Lapacz(Source) : a(1) {}
                };


                static char przec(Dest,int)
                {
                    return 'A';
                }

                static double przec(Lapacz,...)
                {
                    return 12.3;
                }

            };

            template <class Dest,class Sour> struct Przec<Dest*,Sour*> {

                template <class T>
                static char przec(T,int)
                {
                    return 'A';
                }

            };


            template <int arg> struct Cast {

                template <class Sour,class Dest>
                static void make(Dest& d,const Sour& s)
                {
                    d=Dest();
                }
            };

            template <> struct Cast<1> {

                template <class Sour,class Dest>
                static void make(Dest& d,const Sour& s)
                {
                    d=(Dest)s;
                }
            };

        }
    }



	/** \brief Standard caster.
	 *
	 *  Casters are function objects that generate info objects for new-created elements (vertices or edges)
	 *   in methods like Graph::copy, Graph::substitute or methods in class LineGraph, Product and others.
	 *
	 *  The structure overloads call function operator for two parameters. The first parameter is the reference to new-created info object.
	 *  The second parameter is the source info.
	 *  Standard caster tries to simply cast source object on destination object or if this is not possible it calls destination default value (calls empty constructor).
	 *  \wikipath{caster, Get more information about casters.}
	 *  \ingroup DMcaster*/
	struct StdCaster
	{
		typedef StdCaster CastersSelfType;/**<\brief Caster self type, the type defined for each caster.*/

		/** \brief Function call operator.
		 *
		 *  The template overloaded function call operator with two parameters, that uses the inbuilt type cast of \a sour to \a dest.
		 *  It it is impossible \a dest gets its type default value (empty constructor is called).
		 *  \tparam InfoDest the type of destination info object.
		 *  \tparam InfoSour the type of source info object.
		 *  \param dest the reference to the destination new-created info object.
		 *  \param sour the source info object, casted if possible. */
		template< class InfoDest, class InfoSour >
			void operator()( InfoDest &dest, InfoSour sour )
			{
			    //dest = (InfoDest)sour;
                Koala::Privates::KonwTestSpace::Cast<
                        sizeof(Koala::Privates::KonwTestSpace::Przec<InfoDest,InfoSour>::przec(sour,12))
                    >::make(dest,sour);
			}
	};

	/** \brief Generating function for StdCaster.
	 *
	 *  \wikipath{caster, Get more information about casters.}
	 *  \return StdCaster function object that implements overloaded template function call operator which
	 *  tries to cast source info object to destination info object or if it is impossible destination gets the default value.
	 *  \related StdCaster
	 *  \ingroup DMcaster*/
	inline StdCaster stdCast() { return StdCaster(); }

	/** \brief Standard hard caster.
	 *
	 *  Casters are function objects that generate info objects for new-created elements (vertices or edges)
	 *   in methods like Graph::copy, Graph::substitute or methods in class LineGraph, Product and others.
	 *
	 *  The structure overloads call function operator for two parameters. The first parameter is the reference to new-created info object.
	 *  The second parameter is the source info.
	 *  Hard caster tries to simply cast source object on destination object. However, such cast may cause compilation error.
	 *  \wikipath{caster, Get more information about casters.}
	 *  \warning This method may cause compilation error, if the cast is not possible.
	 *  \ingroup DMcaster*/
	struct HardCaster
	{
		typedef HardCaster CastersSelfType;/**<\brief Caster self type, the type defined for each caster.*/

		/** \brief Function call operator.
		 *
		 *  The template overloaded function call operator with two parameters, that uses the inbuilt type cast of \a sour to \a dest.
		 *  It it is impossible the method may cause compilation error.
		 *  \tparam InfoDest the type of destination info object.
		 *  \tparam InfoSour the type of source info object.
		 *  \param dest the reference to the destination new-created info object.
	 	 *  \param sour the source info object, casted if possible. */
		template< class InfoDest, class InfoSour >
			void operator()( InfoDest &dest, InfoSour sour )
			{
			    dest = (InfoDest)sour;
			}
	};

	/** \brief Generating function for HardCaster.
	 *
	 *  \wikipath{caster, Get more information about casters.}
	 *  \return HardCaster function object that implements overloaded template function call operator which
	 *  tries to cast source info object to destination info object.
	 *  \warning Generated caster may cause compilation error, if the cast from source to destination is impossible.
	 *  \related HardCaster
	 *  \ingroup DMcaster*/
	inline HardCaster hardCast() { return HardCaster(); }


	/** \brief No cast caster.
	 *
	 *  Casters are function objects that generate info objects for new-created elements (vertices or edges)
	 *   in methods like Graph::copy, Graph::substitute or methods in class LineGraph, Product and others.
	 *
	 *  The structure overloads call function operator for two and three parameters. The first parameter is the reference to new-created info object.
	 *  The remaining parameters are the source infos. However, NoCastCaster ignores the source infos and sets up the destination for its type
	 *  default value (empty constructor is called).
	 *  \wikipath{caster, Get more information about casters.}
	 *  \ingroup DMcaster*/
	struct NoCastCaster
	{
		typedef NoCastCaster CastersSelfType;/**<\brief Caster self type, the type defined for each caster.*/

		/** \brief Function call operator.
		 *
		 *  The template overloaded function call operator with two parameters. However, the second parameter is ignored and the destination
		 *  gets its type default value.
		 *  \tparam InfoDest the type of destination info object.
		 *  \tparam InfoSour the type of source info object.
		 *  \param dest the reference to the destination new-created info object.
	 	 *  \param sour the source info object, ignored. */
		template< class InfoDest, class InfoSour >
			void operator()( InfoDest &dest, InfoSour sour ) { dest = InfoDest(); }

		/** \brief Function call operator.
		 *
		 *  The template overloaded function call operator with three parameters. However, the second and the third parameter are ignored and the destination
		 *  gets its type default value.
		 *  \tparam InfoDest the type of destination info object.
		 *  \tparam InfoSour1 the type of first source info object.
		 *  \tparam InfoSour2 the type of second source info object.
		 *  \param dest the reference to the destination new-created info object.
	 	 *  \param sour1 the first source info object, ignored.
		 *  \param sour2 the second source info object, ignored. */
		template< class InfoDest, class InfoSour1, class InfoSour2 >
			void operator()( InfoDest &dest, InfoSour1 sour1, InfoSour2 sour2 ) { dest = InfoDest(); }
	};

	/** \brief Generating function for NoCastCaster.
	 *
	 *  \wikipath{caster, Get more information about casters.}
	 *  \return NoCastCaster function object that implements overloaded template function call operator for two and three parameters which
	 *  ignores source info objects and sets destination into object with its type default empty value (the empty constructor is called).
	 *  \param arg only false values are allowed.
	 *  \related NoCastCaster
	 *  \ingroup DMcaster*/
	inline NoCastCaster stdCast( bool arg );

	/** \brief Generating function for NoCastCaster.
	 *
	 *  \wikipath{caster, Get more information about casters.}
	 *  \return NoCastCaster function object that implements overloaded template function call operator for two and three parameters which
	 *  ignores source info objects and sets destination into object with its type default empty value (the empty constructor is called).
	 *  \related NoCastCaster
	 *  \ingroup DMcaster*/
	inline NoCastCaster valCast() { return NoCastCaster(); }

	/** \brief Functor caster.
	 *
	 *  Casters are function objects that generate info objects for new-created elements (vertices or edges)
	 *   in methods like Graph::copy, Graph::substitute or methods in class LineGraph, Product and others.
	 *
	 *  The structure overloads call function operator for two and three parameters. The first parameter is the reference to new-created info object.
	 *  The remaining parameters are the source infos. Function object defined in constructor takes source info objects
	 *  and returns new info object that is casted on destination object type.
	 *  \wikipath{caster, Get more information about casters.}
	 *  \ingroup DMcaster*/
	template< class Fun > struct ObjCaster
	{
		typedef ObjCaster< Fun > CastersSelfType;/**<\brief Caster self type, the type defined for each caster.*/

		mutable Fun functor;/**< \brief the functor defined in constructor.*/

		/**\brief Constructor.
		 *
		 *  The constructor assigns the value to \a functor.*/
		ObjCaster( Fun afun = Fun() ): functor( afun ) { }

		/** \brief Function call operator.
		 *
		 *  The template overloaded function call operator with two parameters. Source info object is sent to function object \a funktor the result
		 *  is casted on destination type and save in \a dest.
		 *  \tparam InfoDest the type of destination info object.
		 *  \tparam InfoSour the type of source info object.
		 *  \param dest the reference to the destination new-created info object.
		 *  \param sour the source info object, sent to \a funkotr. */
		template< class InfoDest, class InfoSour >
			void operator()( InfoDest &dest, InfoSour sour ) { dest = (InfoDest)functor( sour ); }

		/** \brief Function call operator.
		 *
		 *  The template overloaded function call operator with three parameters. Source info objects are sent to function object \a funktor. The returned value
		 *  is casted on destination type and save in \a dest.
		 *  \tparam InfoDest the type of destination info object.
		 *  \tparam InfoSour1 the type of the first source info object.
		 *  \tparam InfoSour2 the type of the second source info object.
		 *  \param dest the reference to the destination new-created info object.
		 *  \param sour1 the first source info object, sent to \a funkotr.
		 *  \param sour2 the second source info object, sent to \a funkotr.*/
		template< class InfoDest, class InfoSour1, class InfoSour2 > void
			operator()( InfoDest &dest, InfoSour1 sour1, InfoSour2 sour2 ) { dest = (InfoDest)functor( sour1,sour2 ); }
	};

	/** \brief Generating function for ObjCaster.
	 *
	 *  \wikipath{caster, Get more information about casters.}
	 *  \tparam Funkotr the type of object function.
	 *  \param f the function object that generates new info object basing on source infos.
	 *  \return ObjCaster function object that implements overloaded template function call operator for two and three parameters.
	 *  The returned functor generates new info object basing on its source info passed to object function \a f.
	 *  \related ObjCaster
	 *  \ingroup DMcaster*/
	template< class Funktor > ObjCaster< Funktor > stdCast(Funktor f) { return ObjCaster< Funktor >(f); }

	/** \brief Common value caster.
	 *
	 *  Casters are function objects that generate info objects for new-created elements (vertices or edges)
	 *   in methods like Graph::copy, Graph::substitute or methods in class LineGraph, Product and others.
	 *
	 *  The structure overloads call function operator for two and three parameters. The first parameter is the reference to new-created info object.
	 *  The remaining parameters are the source infos. However, ValueCaster ignores them and sets the destination info with common value \a val defined in constructor.
	 *  That value is casted on destination info object type. Hence, such cast must be possible.
	 *  \wikipath{caster, Get more information about casters.}
	 *  \ingroup DMcaster*/
	template< class T > struct ValueCaster
	{
		typedef ValueCaster< T > CastersSelfType;/**<\brief Caster self type, the type defined for each caster.*/

		T val;/**<\brief the fixed value set up in constructor, assigned to each new-created info object.*/

		/** \brief Constructor.
		 *
		 *  Sets value \a val up.*/
		ValueCaster( T arg = T() ): val( arg ) { }

		/** \brief Call function operator.
		 *
		 *  The overloaded call function operator with two parameters. The method casts the \a val to \a dest.
		 *  Parameter \a sour is ignored.
		 *  \param dest the reference to the destination info object.
		 *  \param sour the source object (ignored). */
		template< class InfoDest, class InfoSour >
			void operator()( InfoDest &dest, InfoSour sour ) { dest = (InfoDest)val; }

		/** \brief Call function operator.
		 *
		 *  The overloaded call function operator with three parameters. The method casts the \a val to \a dest.
		 *  Parameters \a sour1 and \a sour2 are ignored.
		 *  \param dest the reference to the destination info object.
		 *  \param sour1 the first source object (ignored).
		 *  \param sour2 the second source object (ignored). */
		template< class InfoDest, class InfoSour1, class InfoSour2 >
			void operator()( InfoDest &dest, InfoSour1 sour1, InfoSour2 sour2 ) { dest = (InfoDest)val; }
	};

	/** \brief Generating function for fixed value caster (ObjCaster).
	 *
  	 *  \wikipath{caster, Get more information about casters.}
	 *  \tparam T the type of common value.
	 *  \param arg the value assigned to each info.
	 *  \return ValueCaster function object, that assigns constant value \a arg to each new-created info object.
	 *  \related ValueCaster
	 *  \ingroup DMcaster*/
	template< class T > ValueCaster< T > valCast( T arg ) { return ValueCaster< T >( arg ); }

	/** \brief No link.
	 *
	 *  Methods like Graph::copy, Graph::substitute or methods in class LineGraph, Product and others generate new elements like edges and vertices.
	 *  Sometimes user may want to preserve the information from where new elements origin. For such purposes linkers are designed.
	 *  The linkers are object functions with overloaded call function operator. The first parameter of that function is destination object the second is
	 *  source object.
	 *  This is auxiliary single direction linker, that may be used in bidirectional linker Std2Linker.
	 *  Std1NoLinker is a linker that makes no link. It should be used in cases when user doesn't need a link and doesn't want to create one but function in Koala require some.
	 *
	 *  \ingroup DMlinker */
	struct Std1NoLinker
	{
		/** \brief Constructor.
		 *
		 * \param arg always false.*/
		Std1NoLinker( bool arg = false ) { koalaAssert( !arg,ExcBase ); }

		/** \brief Function call operator.
		 *
		 *  Does nothing.
		 *  \param wsk the pointer to destination object (ignored).
		 *  \param w the pointer to source object (ignored).*/
		template< class Dest, class Sour > void operator()( Dest *wsk, Sour *w ) { }
	};

	/** \brief Object info one direction linker.
	 *
	 *  Methods like Graph::copy, Graph::substitute or methods in class LineGraph, Product and others generate new elements like edges and vertices.
	 *  Sometimes user may want to preserve the information from where new elements origin. For such purposes linkers are designed.
	 *  The linkers are object functions with overloaded call function operator. The first parameter of that function is destination object, the second is
	 *  source object.
	 *
	 *  Std1FieldLinker uses the member of the destination info object to point to object of origin.
	 *  The modified field of the object info is set in constructor via pointer to member.
	 *  This is auxiliary single direction linker, that may be used in bidirectional linker Std2Linker.
	 *  \wikipath{linker, Get more data about linkers.}
	 *  \ingroup DMlinker */
	template< class Info, class T > struct Std1FieldLinker
	{
		T Info:: *pt;/**< \brief Pointer to member in new element info object that keeps a pointer to origin .*/

		/** \brief Constructor.
		 *
		 *  Sets \a pt as a pointer to member in new element info object that stores pointer to origin.*/
		Std1FieldLinker( T Info:: *awsk = 0): pt( awsk ) { }

		/** \brief Function call operator.
		 *
		 *  The function makes connection between the source \a w and the destination \a wsk. In a result the info of \a wsk is modified.*/
		template< class Dest, class Sour > void operator()( Dest *wsk, Sour *w )
			{ if (pt && wsk) wsk->info.*pt = (T) w; }
	};

	/** \brief Associative array one direction linker.
	 *
	 *  Methods like Graph::copy, Graph::substitute or methods in classes LineGraph, Product and others generate new elements edges or vertices.
	 *  Sometimes user may want to preserve the information from where new elements origin. For such purposes linkers are designed.
	 *  The linkers are object functions with overloaded call function operator. The first parameter of that function is pointer to destination object, the second is
	 *  pointer to source object.
	 *
	 *  Std1AssocLinker function object is equipped with external associative container, which keeps information about the link. If this linker is used
	 *  for each new-created element there is a new key (pointer to new element) inserted into associative array where the mapped value is the pointer to the origin.
	 *  Since the map is external and it is not copied, user should keep it valid and up to date.
	 *  This is auxiliary single direction linker, that may be used in bidirectional linker Std2Linker.
	 *  \wikipath{linker, Get more data about linkers.}
	 *  \ingroup DMlinker */
	template< class Map > struct Std1AssocLinker
	{
		Map &map;/**< \brief The associative container (pointer to destination -> pointer to source).*/

		/** \brief Constructor.
		 *
		 *  Sets up the argument \a map value (to the parameter \a amap).*/
		Std1AssocLinker( Map &amap ): map( amap ) { }

		/** \brief Function call operator.
		 *
		 *  The template function makes connection between the source \a w and the destination \a wsk.
		 *  The operator adds a pair to associative array \a amap in which \a wsk is a key and \a w is a mapped value.
		 *  \tparam Dest the type of destination object.
		 *  \tparam Sour the type of source object.
		 *  \param wsk pointer to destination element (PEdge or PVert).
		 *  \param w the pointer to source element (PEdge or PVert).*/
		template< class Dest, class Sour > void operator()( Dest *wsk, Sour *w ) { if (wsk) map[wsk] = w; }
	};

namespace Privates {

	struct Std1PtrLinker
	{
		/** \brief Function call operator.
		 *
		 *  The function makes connection between the source \a w and the destination \a wsk. Create an element in \a amap which associates the key \a wsk with the mapped value \a w.*/
		template< class Dest, class Sour > void operator()( Dest *wsk, Sour *w ) { if (wsk) wsk->info = w; }
	};
}

	/** \brief Bidirectional linker .
	 *
	 *  This linker joins two linkers in order to create two way connection.
	 *  \tparam Link1 the type of linker for destination -> source connection.
	 *  \tparam Link2 the type of linker source -> destination connection.
	 *  \wikipath{linker, Get more data about linkers.}
	 *  \ingroup DMlinker */
	template< class Link1, class Link2 > struct Std2Linker
	{
		typedef Std2Linker< Link1, Link2> LinkersSelfType;

		mutable Link1 dest2sour;/**< \brief The fist linker (destination -> source).*/
		mutable Link2 sour2dest;/**< \brief The second linker (source -> destination).*/

		Link1 &first() { return dest2sour; } /**<\brief Get first linker (destination -> source).*/
		Link2 &second() { return sour2dest; }/**<\brief Get second  linker (source -> destination).*/

		/** \brief Constructor.
		 *
		 *  Sets up the linkers \a dest2sour (\a al1) and \a sour2dest (\a al2).*/
		Std2Linker( Link1 al1, Link2 al2 ): dest2sour( al1 ), sour2dest( al2 ) { }

		/** \brief Function call operator.
		 *
		 *  The function makes connection between the source \a w and the destination \a wsk and another way round.
		 *  The operator uses methods lining elements delivered by linkers set up in constructor.*/
		template< class Dest, class Sour > void operator()( Dest *wsk, Sour *w );
	};

	/** \brief Generating function of no linker (Std1NoLinker).
	 *
	 *  \param a1 always false
	 *  \return linker Std1NoLinker that does not make a connection.
	 *  \related Std1NoLinker
	 *  \ingroup DMlinker */
	inline Std1NoLinker stdLink( bool a1 ) { return Std1NoLinker( a1 ); }

	/** \brief Generating function of one direction info linker (Std1FieldLinker).
	 *
	 *  The function generates and returns the one direction linker object the uses the info object of destination elements.
	 *  \param awsk1 the pointer to the member of info object.
	 *  \return Std1FieldLinker function object that is able to connect two elements using member of destination info object..
	 *  \related Std1FieldLinker
	 *  \ingroup DMlinker */
	template< class Info1, class T1 >
		Std1FieldLinker< Info1,T1 > stdLink( T1 Info1:: *awsk1 ) { return Std1FieldLinker< Info1,T1 >( awsk1 ); }

	/** \brief Generating function of standard one direction linker (Std1AssocLinker).
	 *
	 *  The function generates and returns the one direction linker object the uses external associative array \a tab1.
	 *  \param tab1 the reference to the associative array that keeps all the connections.
	 *  \return the linker function object that is able to connect two elements via associative array.
	 *  \related Std1AssocLinker
	 *  \ingroup DMlinker     */
	template< class Map1 >
		Std1AssocLinker< Map1 > stdLink( Map1 &tab1 ) { return Std1AssocLinker< Map1 >( tab1 ); }

	/** \brief Generating function for no linker based on Std2Linker.
	 *
	 *  The function generates two directional link, however both links are dummy.
	 *  Boolean parameters take only false value, then the link is not created.
	 *  \related Std2Linker
	 *  \related Std1NoLinker
	 *  \ingroup DMlinker */
	inline Std2Linker< Std1NoLinker,Std1NoLinker > stdLink( bool a1, bool a2 );

	/** \brief Generating function of one way field linker based on Std2Linker.
	 *
	 *  \param a1 Boolean parameter take only false,
	 *  \param awsk pointer to member in source info object where pointer to destination object is stored.
	 *  \return the linker with one way connection source -> destination, using field pointed by \a awsk in source info object.
	 *  \related Std2Linker
	 *  \related Std1NoLinker
	 *  \related Std1FieldLinker
	 *  \ingroup DMlinker     */
	template< class Info,class T >
		Std2Linker< Std1NoLinker,Std1FieldLinker< Info,T > > stdLink( bool a1, T Info:: *awsk );

	/** \brief Generating function of one way linker based on Std2Linker.
	 *
	 *  \param a1 Boolean parameter take only false,
	 *  \param tab associative container assigning  destination to source (pairs (sour, dest)).
	 *  \return the linker with one way connection source to destination, based on associative container.
	 *  \related Std2Linker
	 *  \related Std1NoLinker
	 *  \related Std1AssocLinker
	 *  \ingroup DMlinker     */
	template< class Map >
		Std2Linker< Std1NoLinker,Std1AssocLinker< Map > > stdLink( bool a1, Map &tab );

	/** \brief Generating function for one way linker based on Std2Linker.
	 *
	 *  \param awsk1 pointer to member
	 *  \param a2 Boolean parameter take only false,
	 *  \return the linker with one way connection destination to source, using field pointed by \a awsk1 in destination info object.
	 *  \related Std2Linker
	 *  \related Std1FieldLinker
	 *  \related Std1NoLinker
	 *  \ingroup DMlinker     */
	template< class Info1, class T1 >
		Std2Linker< Std1FieldLinker< Info1,T1 >,Std1NoLinker > stdLink( T1 Info1:: *awsk1, bool a2 );

	/** \brief Generating function for two way linker based on Std2Linker.
	 *
	 *  \param awsk1 the pointer to member in destination info
	 *  \param awsk2 the pointer to member in source info.
	 *  \return the linker with two way connection, using field pointed by \a awsk1 in destination info object and the field pointed by \a awsk2 in source info object.
	 *  \related Std2Linker
	 *  \related Std1FieldLinker
	 *  \ingroup DMlinker     */
	template< class Info1, class T1, class Info, class T >
		Std2Linker< Std1FieldLinker< Info1,T1 >,Std1FieldLinker< Info,T > > stdLink( T1 Info1:: *awsk1, T Info:: *awsk );

	/** \brief Generating function for two way linker based on Std2Linker.
	 *
	 *  \param awsk1 the pointer to member in destination info
	 *  \param tab the associative container assigning destination to source.
	 *  \return the linker with two way connection, using field pointed by \a awsk1 in destination info object (storing pointer to source)
	 *  and associative container (pairs (sour,dest)).
	 *  \related Std2Linker
	 *  \related Std1FieldLinker
	 *  \related Std1AssocLinker
	 *  \ingroup DMlinker     */
	template< class Info1, class T1, class Map >
		Std2Linker< Std1FieldLinker< Info1,T1 >,Std1AssocLinker< Map > > stdLink( T1 Info1:: *awsk1, Map &tab);

	/** \brief Generating function of one way linker based on Std2Linker.
	 *
	 *  \param a2 boolean parameter take only false,
	 *  \param tab1 associative container assigning  source to destination (pairs (sour, dest)).
	 *  \return the linker with one way connection destination to source, basing on associative container.
	 *  \related Std2Linker
	 *  \related Std1AssocLinker
	 *  \related Std1NoLinker
	 *  \ingroup DMlinker     */
	template< class Map1 >
		Std2Linker< Std1AssocLinker< Map1 >,Std1NoLinker > stdLink( Map1 &tab1, bool a2 );

	/** \brief Generating function for two way linker based on Std2Linker.
	 *
	 *  \param tab1 the associative container assigning source to destination.
	 *  \param awsk the pointer to member in source info object where pointer to destination is stored.
	 *  \return the linker with two way connection, using associative container and field pointed by \a awsk in the source info object.
	 *  \related Std2Linker
	 *  \related Std1AssocLinker
	 *  \related Std1FieldLinker
	 *  \ingroup DMlinker     */
	template< class Map1, class Info, class T >
		Std2Linker< Std1AssocLinker< Map1 >,Std1FieldLinker< Info,T > > stdLink( Map1 &tab1, T Info:: *awsk );

	/** \brief Generating function for two way linker based on Std2Linker.
	 *
	 *  \param tab1 the associative container assigning source to destination (pairs (dest, sour)).
	 *  \param tab the associative container assigning destination to source  (pairs (sour, dest)).
	 *  \return the linker with two way connection, based on two associative containers.
	 *  \related Std2Linker
	 *  \related Std1AssocLinker
	 *  \ingroup DMlinker     */
	template< class Map1, class Map >
		Std2Linker< Std1AssocLinker< Map1 >,Std1AssocLinker< Map > > stdLink( Map1 &tab1, Map &tab );

	/**\brief Make pair of choosers.
	 *
	 * Overloaded operator& allows to create easily a std::pair of choosers \a a and \a b.*
	 * \ingroup DMchooser */
	template <class  Ch1, class Ch2> std::pair< typename Ch1::ChoosersSelfType,typename Ch2::ChoosersSelfType >
		operator&( Ch1 a, Ch2 b )
		{
			return std::pair< typename Ch1::ChoosersSelfType,typename Ch2::ChoosersSelfType >(a,b);
		}


	/**\brief Make pair of casters.
	 *
	 * Overloaded operator& allows to create easily a std::pair of casters \a a and \a b.
	 *
	 * \related NoCastCaster
	 * \related ValueCaster
	 * \related HardCaster
	 * \related StdCaster
	 * \related ObjCaster
	 * \ingroup DMcaster */
	template <class  Ch1, class Ch2> std::pair< typename Ch1::CastersSelfType,typename Ch2::CastersSelfType >
		operator&( Ch1 a, Ch2 b )
		{
			return std::pair< typename Ch1::CastersSelfType,typename Ch2::CastersSelfType >(a,b);
		}

	/**\brief Make pair of linkers.
	 *
	 * Overloaded operator& allows to create easily a pair of std::linkers \a a and \a b.
	 * \related Std1NoLinker
	 * \related Std1FieldLinker
	 * \related Std1AssocLinker
	 * \ingroup DMlinker */
	template <class  Ch1, class Ch2> std::pair< typename Ch1::LinkersSelfType,typename Ch2::LinkersSelfType >
		operator&( Ch1 a, Ch2 b )
		{
			return std::pair< typename Ch1::LinkersSelfType,typename Ch2::LinkersSelfType >(a,b);
		}


#include "defs.hpp"
}

#endif
