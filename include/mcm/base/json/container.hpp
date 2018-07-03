#ifndef MV_JSON_CONTAINER_HPP_
#define MV_JSON_CONTAINER_HPP_

namespace mv
{

    namespace json
    {

        class Container
        {
            
        public:

            class Index
            {

            public:

                virtual ~Index() = 0;

            };

        private:

        public:

            virtual ~Container() = 0;

        }

    };

}

#endif // MV_JSON_CONTAINER_