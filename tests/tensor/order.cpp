#include "gtest/gtest.h"
#include "include/mcm/tensor/order.hpp"

TEST(order, col_major0d)
{
    mv::Order order(mv::OrderType::ColumnMajor);
    mv::Shape s(0);
    ASSERT_ANY_THROW(order.isFirstContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.isLastContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.previousContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.nextContiguousDimensionIndex(s, 0));
}

TEST(order, col_major1d)
{
    mv::Order order(mv::OrderType::ColumnMajor);
    mv::Shape s({3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, col_major2d)
{
    mv::Order order(mv::OrderType::ColumnMajor);
    mv::Shape s({2, 3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, col_major3d)
{
    mv::Order order(mv::OrderType::ColumnMajor);
    mv::Shape s({2, 3, 5});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, col_major4d)
{
    mv::Order order(mv::OrderType::ColumnMajor);
    mv::Shape s({3, 3, 5, 7});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, row_major0d)
{
    mv::Order order(mv::OrderType::RowMajor);
    mv::Shape s(0);
    ASSERT_ANY_THROW(order.isFirstContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.isLastContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.previousContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.nextContiguousDimensionIndex(s, 0));
}


TEST(order, row_major1d)
{
    mv::Order order(mv::OrderType::RowMajor);
    mv::Shape s({3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, row_major2d)
{
    mv::Order order(mv::OrderType::RowMajor);
    mv::Shape s({2, 3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, row_major3d)
{
    mv::Order order(mv::OrderType::RowMajor);
    mv::Shape s({2, 3, 5});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, row_major4d)
{
    mv::Order order(mv::OrderType::RowMajor);
    mv::Shape s({3, 3, 5, 7});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar0d)
{
    mv::Order order(mv::OrderType::ColumnMajorPlanar);
    mv::Shape s(0);
    ASSERT_ANY_THROW(order.isFirstContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.isLastContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.previousContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.nextContiguousDimensionIndex(s, 0));
}

TEST(order, planar1d)
{
    mv::Order order(mv::OrderType::ColumnMajorPlanar);
    mv::Shape s({3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar2d)
{
    mv::Order order(mv::OrderType::ColumnMajorPlanar);
    mv::Shape s({2, 3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar3d)
{
    mv::Order order(mv::OrderType::ColumnMajorPlanar);
    mv::Shape s({2, 3, 5});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar4d)
{
    mv::Order order(mv::OrderType::ColumnMajorPlanar);
    mv::Shape s({3, 3, 5, 7});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar20d)
{
    mv::Order order(mv::OrderType::RowMajorPlanar);
    mv::Shape s(0);
    ASSERT_ANY_THROW(order.isFirstContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.isLastContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.previousContiguousDimensionIndex(s, 0));
    ASSERT_ANY_THROW(order.nextContiguousDimensionIndex(s, 0));
}

TEST(order, planar21d)
{
    mv::Order order(mv::OrderType::RowMajorPlanar);
    mv::Shape s({3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar22d)
{
    mv::Order order(mv::OrderType::RowMajorPlanar);
    mv::Shape s({2, 3});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar23d)
{
    mv::Order order(mv::OrderType::RowMajorPlanar);
    mv::Shape s({2, 3, 5});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

TEST(order, planar24d)
{
    mv::Order order(mv::OrderType::RowMajorPlanar);
    mv::Shape s({3, 3, 5, 7});

    try
    {
        for(int i = order.firstContiguousDimensionIndex(s); i != -1; i = order.nextContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for(int i = order.lastContiguousDimensionIndex(s); i != -1; i = order.previousContiguousDimensionIndex(s,i))
        {
            std::cout << i;
            if(order.isFirstContiguousDimensionIndex(s, i))
                std::cout << " - Is the first contiguos dimension";
            if(order.isLastContiguousDimensionIndex(s, i))
                std::cout << " - Is the last contiguos dimension";
            std::cout << std::endl;
        }
   }
   catch(mv::ShapeError err)
   {
        std::cout << "Catched exception " << err.what() << std::endl;
   }
}

