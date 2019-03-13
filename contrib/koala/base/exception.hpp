// ExcBase

namespace Koala
{
	namespace Error
	{
		inline ExcBase::ExcBase( const char *adesc, const char *afile, int aline ): _line(aline)
		{
			std::strncpy( buf,adesc,KOALA_EXCEPTION_BUF_SIZE-2 );
			buf[KOALA_EXCEPTION_BUF_SIZE-2]=0;
			std::strncpy( buf + std::strlen( buf ) + 1,afile,KOALA_EXCEPTION_BUF_SIZE-2-std::strlen( buf ));
			buf[KOALA_EXCEPTION_BUF_SIZE-1]=0;
		}
	}
}
