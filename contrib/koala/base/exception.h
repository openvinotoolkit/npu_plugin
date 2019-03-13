#ifndef KOALA_EXCEPTION_H
#define KOALA_EXCEPTION_H

/** \file exception.h
 *  \brief Exception variants and handling (included automatically).
 */

#include <cassert>
#include <cstdlib>
#include <cstring>

/**\def KOALA_EXCEPTION_BUF_SIZE 
 * \brief Size of exception text buffer. 
 *  \ingroup DMexception */
#ifndef KOALA_EXCEPTION_BUF_SIZE
	#define KOALA_EXCEPTION_BUF_SIZE 150
#endif

/** \def KOALA_DONT_CHECK_ERRORS 
 *  \brief Macro switching of exception testing.
 *  \ingroup DMexception */
#if defined(NDEBUG)
	#define KOALA_DONT_CHECK_ERRORS
#endif

/** \def koalaAssert( descr,type )
 *  \brief Koala macro for throwing exceptions.
 *  
 *  The macro takes token \a descr as parameter and type of Error \a type. Macro also uses information about from where (file name and line number) the exception is thrown.
 *  \ingroup DMexception */
#if defined(KOALA_DONT_CHECK_ERRORS)
	#define koalaAssert( descr,type ) {}
#else
	#define koalaAssert( descr,type ) { if (!(descr)) throw Koala::Error::type( #descr,__FILE__,__LINE__ ); }
#endif

namespace Koala
{
	/** \brief Exceptions */
	namespace Error
	{
		/** \brief Exception base.
		 *
		 * The base class for Koala exceptions.
		 * \ingroup DMexception */
		class ExcBase
		{
			char buf[KOALA_EXCEPTION_BUF_SIZE];
			int _line;

		public:
			/** \brief Constructor
			 *
			 * \param adesc Error description
			 * \param afile file name where the exception is thrown.
			 * \param aline the line number of the exception occurrence.*/
			inline ExcBase( const char *adesc = "", const char *afile = "", int aline = -1);

			/**\brief Get exception occurrence line number.*/
			inline int line() const
				{ return _line; }
			/**\brief Get exception description.*/
			inline const char *descr() const
				{ return buf; }
			/**\brief Get source file name where the exception is thrown.*/
			inline const char *file() const
				{ return buf + std::strlen( buf ) + 1; }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ExcBase"; }
		};

		/** \brief Wrong argument exception.
		 *
		 * \ingroup DMexception */
		class ExcWrongArg: virtual public ExcBase
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ExcWrongArg( const char *adesc = "", const char *afile = "", int aline = -1):
				ExcBase( adesc,afile,aline )
					{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ExcWrongArg"; }
		};

		/** \brief NULL vertex pointer exception.
		 *
		 * \ingroup DMexception */
		class ExcNullVert: public ExcWrongArg
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ExcNullVert(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ExcNullVert"; }
		};

		/** \brief NULL edge pointer exception.
		 *
		 * \ingroup DMexception */
		class ExcNullEdge: public ExcWrongArg
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ExcNullEdge(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ExcNullEdge"; }
		};

		/** \brief Wrong vertex - edge connection exception.
		 *
		 * \ingroup DMexception */
		class ExcWrongConn: public ExcWrongArg
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ExcWrongConn(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ExcWrongConn"; }
		};

		/** \brief Wrong edge type exception.
		 *
		 * \ingroup DMexception */
		class ExcWrongMask: public ExcWrongArg
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ExcWrongMask(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ExcWrongMask"; }
		};

		/** \brief Base class for exceptions in containers.
		 *
		 * \ingroup DMexception */
		class ContExc: virtual public ExcBase
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ContExc(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ContExc"; }
		};

		/** \brief  Container method incorrect argument exception.
		 *
		 * \ingroup DMexception */
		class ContExcWrongArg: public ContExc, public ExcWrongArg
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ContExcWrongArg(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ContExcWrongArg"; }
		};

		/** \brief  Container overflow exception.
		 *
		 * \ingroup DMexception */
		class ContExcFull: public ContExc
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ContExcFull(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ContExcFull"; }
		};


		/**\brief Pool SimplArrPool Destructor exception. Not all object were deallocated.
		 *
		 * \ingroup DMexception*/
		class ContExcPoolNotEmpty: public ContExc
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ContExcPoolNotEmpty(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ContExcPoolNotEmpty"; }
		};


		/** \brief  Container rage outpass exception.
		 *
		 * \ingroup DMexception */
		class ContExcOutpass: public ContExcWrongArg
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline ContExcOutpass(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "ContExcOutpass"; }
		};

		/** \brief Base class for exceptions in graphs.
		 *
		 * \ingroup DMexception */
		class GraphExc: virtual public ExcBase
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline GraphExc(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "GraphExc"; }
		};

		/** \brief Wrong argument of graph structure method exception.
		 *
		 * \ingroup DMexception */
		class GraphExcWrongArg: public GraphExc, public ExcWrongArg
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline GraphExcWrongArg(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "GraphExcWrongArg"; }
		};

		/** \brief Wrong argument of graph structure method (incorrect connection)  exception.
		 *
		 * \ingroup DMexception */
		class GraphExcWrongConn: public GraphExc, public ExcWrongConn
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline GraphExcWrongConn(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "GraphExcWrongConn"; }
		};

		/** \brief Wrong argument of graph structure method (NULL pointer of vertex)  exception.
		 *
		 * \ingroup DMexception */
		class GraphExcNullVert: public GraphExc, public ExcNullVert
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline GraphExcNullVert(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "GraphExcNullVert"; }
		};

		/** \brief Wrong argument of graph structure method (NULL pointer of edge)  exception.
		 *
		 * \ingroup DMexception */
		class GraphExcNullEdge: public GraphExc, public ExcNullEdge
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline GraphExcNullEdge(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "GraphExcNullEdge"; }
		};

		/** \brief Wrong argument of graph structure method (incorrect edge type)  exception.
		 *
		 * \ingroup DMexception */
		class GraphExcWrongMask: public GraphExc, public ExcWrongMask
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline GraphExcWrongMask(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "GraphExcWrongMask"; }
		};

		/** \brief Base class for exceptions in algorithms.
		 *
		 * \ingroup DMexception */
		class AlgExc: virtual public ExcBase
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline AlgExc(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "AlgExc"; }
		};

		/** \brief Wrong argument of algorithm exception.
		 *
		 * \ingroup DMexception */
		class AlgExcWrongArg: public AlgExc, public ExcWrongArg
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline AlgExcWrongArg(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "AlgExcWrongArg"; }
		};

		/** \brief Wrong argument of algorithm exception (NULL vertex pointer).
		 *
		 * \ingroup DMexception */
		class AlgExcNullVert: public AlgExc, public ExcNullVert
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline AlgExcNullVert(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "AlgExcNullVert"; }
		};

		/** \brief Wrong argument of algorithm exception (NULL edge pointer).
		 *
		 * \ingroup DMexception */
		class AlgExcNullEdge: public AlgExc, public ExcNullEdge
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline AlgExcNullEdge(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "AlgExcNullEdge"; }
		};

		/** \brief Wrong argument of algorithm exception (incorrect edge type).
		 *
		 * \ingroup DMexception */
		class AlgExcWrongMask: public AlgExc, public ExcWrongMask
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline AlgExcWrongMask(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "AlgExcWrongMask"; }
		};

		/** \brief Wrong argument of algorithm exception (incorrect connection).
		 *
		 * \ingroup DMexception */
		class AlgExcWrongConn: public AlgExc, public ExcWrongConn
		{
		public:
			/**\copydoc ExcBase::ExcBase*/
			inline AlgExcWrongConn(const char *adesc = "", const char *afile = "", int aline = -1) :
				ExcBase( adesc,afile,aline )
				{ }
			/**\brief Get exception type.*/
			inline const char *type() const
				{ return "GraphExcWrongConn"; }
		};
	}
}

#include "exception.hpp"
#endif
