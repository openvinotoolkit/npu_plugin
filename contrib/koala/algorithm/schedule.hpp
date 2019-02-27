// SchedulingPar

void SchedulingStructs::Schedule::clearMachines()
{
	int m = getMachNo();
	machines.clear();
	machines.resize( m );
}

int SchedulingStructs::Schedule::part( int machNo, int time )
{
    koalaAssert( machNo>=0 && machNo<machines.size(),ContExcOutpass );
	Machine &m = machines[machNo];
	Machine::const_iterator i = std::upper_bound(m.begin(), m.end(), TaskPart(-1, time, time), TaskPart::compareEndTime());
	return i == m.end() ? -1 : (i - m.begin());
}

template< typename IntInserter, typename STDPairOfIntInserter >
void SchedulingStructs::Schedule::taskPartList( SearchStructs::CompStore<IntInserter,STDPairOfIntInserter> out )
{
	int n = 0;
	for( Type::const_iterator i = machines.begin(); i != machines.end(); ++i )
		n += i->size();

	TaskPart LOCALARRAY( tasks, n );

	n = 0;
	for( Type::const_iterator i = machines.begin(); i != machines.end(); ++i )
		for( Machine::const_iterator j = i->begin(); j != i->end(); ++j, ++n )
			tasks[n] = *j, tasks[n].end = (i - machines.begin()), tasks[n].part = (j - i->begin());
	std::sort( tasks,tasks + n, TaskPart::compareIndexAndStartTime() );

	for(int i = 0; i < n; i++)
	{
		if(i == 0 || tasks[i - 1].task != tasks[i].task)
			*out.compIter = i, ++out.compIter;
		*out.vertIter = std::make_pair(tasks[i].end, tasks[i].part), ++out.vertIter;
	}
	*out.compIter = n, ++out.compIter;
}

template< class DefaultStructs > template< typename Comp, typename TaskIterator, typename Iterator >
	int SchedulingPar< DefaultStructs >::sortByComp( TaskIterator begin, TaskIterator end, Iterator out )
{
	typedef std::pair< TaskIterator,int > Pair;

	int n = 0;
	for( TaskIterator i = begin; i != end; ++i ) n++;
	Pair LOCALARRAY( tasks,n );
	for( n = 0; begin != end; ++begin,n++ )
	{
		tasks[n].first = begin;
		tasks[n].second = n;
	}
	DefaultStructs::sort( tasks,tasks + n,Comp() );
	for( int i = 0; i < n; i++,++out ) *out = tasks[i].second;
	return n;
}

template< class DefaultStructs > template< typename TaskIterator >
	int SchedulingPar<DefaultStructs>::CMax( TaskIterator begin, TaskIterator end, const Schedule &schedule )
{
	int ans = 0;
	for( typename Schedule::Type::const_iterator i = schedule.machines.begin(); i != schedule.machines.end(); ++i )
		if (!i->empty() && ans < i->rbegin()->end) ans = i->rbegin()->end;
	return ans;
}

template< class DefaultStructs > template< typename TaskIterator >
	int SchedulingPar< DefaultStructs >::SigmaCi( TaskIterator begin, TaskIterator end, const Schedule &schedule )
{
	int n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ ) ;

	int LOCALARRAY( finish,n );
	for( int i = 0; i < n; i++ ) finish[i] = 0;

	for( typename Schedule::Type::const_iterator i = schedule.machines.begin(); i != schedule.machines.end(); ++i )
		for( typename Schedule::Machine::const_iterator j = i->begin(); j != i->end(); ++j )
			if (finish[j->task] < j->end) finish[j->task] = j->end;

	int ans = 0;
	for( int i = 0; i < n; i++ ) ans += finish[i];
	return ans;
}

template< class DefaultStructs > template< typename TaskIterator >
	int SchedulingPar< DefaultStructs >::SigmaTi( TaskIterator begin, TaskIterator end, const Schedule &schedule )
{
	int n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ );

	int LOCALARRAY( finish,n );
	for( int i = 0; i < n; i++ ) finish[i] = 0;

	for( typename Schedule::Type::const_iterator i = schedule.machines.begin(); i != schedule.machines.end(); ++i )
		for( typename Schedule::Machine::const_iterator j = i->begin(); j != i->end(); ++j )
			if (finish[j->task] < j->end) finish[j->task] = j->end;

	int ans = 0, i = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,i++ )
		ans += finish[i] > iterator->duedate ? finish[i] - iterator->duedate : 0;
	return ans;
}

template< class DefaultStructs > template< typename TaskIterator >
	int SchedulingPar< DefaultStructs >::SigmaUi( TaskIterator begin, TaskIterator end, const Schedule &schedule )
{
	int i, n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator, n++ ) ;

	int LOCALARRAY( finish,n) ;
	for( i = 0; i < n; i++ ) finish[i] = 0;

	for( typename Schedule::Type::const_iterator it = schedule.machines.begin(); it != schedule.machines.end(); ++it )
		for( typename Schedule::Machine::const_iterator j = it->begin(); j != it->end(); ++j )
			if (finish[j->task] < j->end) finish[j->task] = j->end;

	int ans = 0;
	i = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,i++ ) ans += (finish[i] > iterator->duedate);
	return ans;
}

template< class DefaultStructs > template< typename TaskIterator >
	int SchedulingPar< DefaultStructs >::LMax( TaskIterator begin, TaskIterator end,  const Schedule &schedule )
{
	int i, n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ ) ;

	int LOCALARRAY( finish,n );
	for( i = 0; i < n; i++ ) finish[i] = 0;

	for( typename Schedule::Type::const_iterator it = schedule.machines.begin(); it != schedule.machines.end(); ++it )
		for( typename Schedule::Machine::const_iterator j = it->begin(); j != it->end(); ++j )
			if (finish[j->task] < j->end) finish[j->task] = j->end;

	i = 0;
	int ans = std::numeric_limits< int >::min();
	for( TaskIterator iterator = begin; iterator != end; ++iterator, i++ )
	{
		int value = finish[i] - iterator->duedate;
		if (ans < value) ans = value;
	}
	return ans;
}

template< class DefaultStructs > template< typename GraphType, typename TaskIterator, typename TaskWindowIterator >
	int SchedulingPar< DefaultStructs >::critPath( TaskIterator begin, TaskIterator end, const GraphType &DAG,
		TaskWindowIterator schedule )
{


	int n = 0, time = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator ) n++;

	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,Triple< Task< GraphType > > >::Type
		tasks( n );

	for( TaskIterator iterator = begin; iterator != end; ++iterator)
		tasks[iterator->vertex] = Triple< Task< GraphType > >(*iterator);

	// computes order of vertices
	typename GraphType::PVertex LOCALARRAY( vertices,n );
	Koala::DAGAlgsPar<DefaultStructs>::topOrd( DAG,vertices );

	// computes times of earliest possible start
	for( int i = 0; i < n; i++ )
	{
		Triple< Task< GraphType > > &element = tasks[vertices[i]];
		element.start = element.task.release;
	}
	for( int i = 0, stop; i < n; i++ )
	{
		typename GraphType::PVertex v = vertices[i];
		Triple< Task< GraphType > > &first = tasks[v];
		stop = first.start + first.task.length;

		for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirOut ); e; e = DAG.getEdgeNext( v,e,EdDirOut ) )
		{
			Triple< Task< GraphType > > &second = tasks[DAG.getEnd( e,v )];
			if (second.start < stop) second.start = stop;
		}

		if (time < stop) time = stop;
	}

	// computes times of latest possible end
	for( int i = 0; i < n; i++ ) tasks[vertices[i]].finish = time;

	for( int i = n - 1, start; i >= 0; i-- )
	{
		typename GraphType::PVertex v = vertices[i];
		Triple< Task< GraphType > > &first = tasks[v];
		start = first.finish - first.task.length;

		for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirIn ); e; e = DAG.getEdgeNext( v,e,EdDirIn ))
		{
			Triple< Task< GraphType > > &second = tasks[DAG.getEnd( e,v )];
			if (second.finish > start) second.finish = start;
		}
	}

	for( TaskIterator iterator = begin; iterator != end; ++iterator )
	{
		Triple< Task< GraphType > > &element = tasks[iterator->vertex];
		*schedule = TaskWindow( element.task,element.start,element.finish ), ++schedule;
	}

	return time;
}

template< class DefaultStructs > template< typename GraphType, typename TaskIterator >
	bool SchedulingPar< DefaultStructs >::test( TaskIterator begin, TaskIterator end, const GraphType &DAG,
		const Schedule &schedule, bool nonPmtn )
{
	typedef std::pair< TaskPart,int > Pair;

	int parts = 0, n = 0;
	// lengths of tasks are nonnegatve
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ )
		koalaAssert( iterator->length > 0,AlgExcWrongArg );

	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,TaskPart >::Type tasks( n );

	// Two different tasks are not executed simultaneously on one machine
	for( typename Schedule::Type::const_iterator i = schedule.machines.begin();
		i != schedule.machines.end(); parts += i->size(),++i )
		if (!i->empty())
			for( typename Schedule::Machine::const_iterator j = i->begin(), k = j + 1; k != i->end(); ++j,++k )
				if (j->end > k->start) return false;

	if (nonPmtn && n != parts) return false;

	TaskPart LOCALARRAY( result,n );

    SimplArrPool< typename DefaultStructs:: template
		HeapCont< Pair, compareSecondFirst< Pair > >::NodeType> available(parts);
    typename DefaultStructs::template
		HeapCont< Pair,compareSecondFirst< Pair > >::Type pq(&available);

	// The same task is not processed simultaneously on two machines
	for( typename Schedule::Type::const_iterator i = schedule.machines.begin(); i != schedule.machines.end(); ++i )
		for( typename Schedule::Machine::const_iterator j = i->begin(); j != i->end(); ++j )
			pq.push( Pair( *j,j->start ) );

	while(!pq.empty())
	{
		TaskPart first = pq.top().first, &second = result[first.task];

		if (first.part != second.part) return false;
		if (!first.part) second.start = first.start;
		else if (second.end > first.start) return false;

		second.part++;
		second.end = first.end;
		second.task += (first.end - first.start);
		pq.pop();
	}

	n = 0;
	// Every task was processed in 100%
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ )
		if (iterator->release > result[n].start || iterator->length != result[n].task) return false;

	n = 0;
	// prec relation
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ ) tasks[iterator->vertex] = result[n];

	for( typename GraphType::PVertex v = DAG.getVert(); v; v = DAG.getVertNext( v ) )
		for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirOut ); e; e = DAG.getEdgeNext( v,e,EdDirOut ) )
			if (tasks[v].end > tasks[DAG.getEnd( e,v )].start) return false;
	return true;
}

template< class DefaultStructs > template< typename GraphType, typename TaskIterator >
	int SchedulingPar< DefaultStructs >::ls( TaskIterator begin, TaskIterator end, const GraphType &DAG,
		Schedule &schedule )
{
	typedef std::pair< int,int > IntPair;
	typedef std::pair< typename GraphType::PVertex,int > Pair;

	SimplArrPool<typename DefaultStructs::template
        HeapCont< IntPair,compareSecondFirst< IntPair > >::NodeType  > machines(schedule.getMachNo());
	typename DefaultStructs::template
        HeapCont< IntPair,compareSecondFirst< IntPair > >::Type machine(&machines);

	for( int i = 0; i < schedule.getMachNo(); i++) machine.push( std::make_pair( i,0 ) );

	int n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ ) ;

	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,Element< Task< GraphType > > >::Type
		tasks( n );

	SimplArrPool<typename DefaultStructs::template
        HeapCont< Pair,compareSecondFirst< Pair > >::NodeType > alloc(n+1);
    typename DefaultStructs::template
        HeapCont< Pair,compareSecondFirst< Pair > >::Type candidate(&alloc), active(&alloc);

	n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ )
	{
		Element< Task< GraphType > > &element = tasks[iterator->vertex] = Element< Task< GraphType > >( *iterator,n );
		element.degree = DAG.getEdgeNo( iterator->vertex,EdDirIn );

		if (element.degree == 0) candidate.push( Pair( iterator->vertex,element.task.release ) );
	}

	int time = 0, out = 0;
	while (n--)
	{
		if (time < machine.top().second) time = machine.top().second;
		int machNo = machine.top().first;
		machine.pop();

		if (active.empty() && time < candidate.top().second) time = candidate.top().second;
		while (!candidate.empty() && candidate.top().second <= time)
		{
			active.push( Pair( candidate.top().first,tasks[candidate.top().first].index ) );
			candidate.pop();
		}

		typename GraphType::PVertex u, v = active.top().first;

		int stop = time + tasks[v].task.length;
		schedule.machines[machNo].push_back( TaskPart( active.top().second,time,stop ) );
		machine.push( std::make_pair( machNo,stop ) );
		active.pop();

		for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirOut ); e; e = DAG.getEdgeNext( v,e,EdDirOut ) )
		{
			Element< Task< GraphType > > &second = tasks[u = DAG.getEdgeEnd( e,v )];
			if (second.priority < stop) second.priority = stop;

			if (--second.degree == 0) candidate.push( Pair( u,std::max( second.priority,second.task.release ) ) );
		}

		if (out < stop) out = stop;
	}

	return out;
}

template< class DefaultStructs > template< typename GraphType, typename TaskIterator >
	int SchedulingPar< DefaultStructs >::coffmanGraham( TaskIterator begin, TaskIterator end, const GraphType &DAG,
		Schedule &schedule )
{
	koalaAssert( schedule.getMachNo() == 2,AlgExcWrongArg );
	typedef std::list< typename GraphType::PVertex > VertexList;

	VertexList vertices;

	int n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator ) n++;
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,std::pair< Task< GraphType >,int > >::Type
		tasks( n );

	std::list< int > LOCALARRAY( candidates,n );

	n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ )
		tasks[iterator->vertex] = std::make_pair( *iterator,n ), vertices.push_back( iterator->vertex );

	int label = 0, i = n;
	Task< GraphType > LOCALARRAY( order,n );
	while (!vertices.empty())
	{
		VertexList active;
		for( typename VertexList::iterator iterator = vertices.begin(); iterator != vertices.end(); )
		{
			if (DAG.getEdgeNo( *iterator,EdDirOut ) == candidates[tasks[*iterator].second].size())
			{
				active.push_back( *iterator );
				iterator = vertices.erase( iterator );
			}
			else
				 ++iterator;
		}

		while (!active.empty())
		{
			typename VertexList::iterator smallest = active.begin();
			int index = tasks[*smallest].second;
			for( typename VertexList::iterator iterator = ++(active.begin()); iterator != active.end(); ++iterator )
			{
				std::list< int > &next = candidates[tasks[*iterator].second];
				if (!std::lexicographical_compare( candidates[index].begin(),candidates[index].end(),next.begin(),next.end() ))
					smallest = iterator, index = tasks[*smallest].second;
			}

			for( typename GraphType::PEdge e = DAG.getEdge( *smallest,EdDirIn ); e; e = DAG.getEdgeNext( *smallest,e,EdDirIn ) )
				candidates[tasks[DAG.getEnd( e,*smallest )].second].push_front( label );

			order[--i] = tasks[*smallest].first, active.erase( smallest ), label++;
		}
	}

	int out = SchedulingPar< DefaultStructs >::ls( order,order + n,DAG,schedule );
	for( typename Schedule::Type::iterator i = schedule.machines.begin(); i != schedule.machines.end(); ++i )
		for( typename Schedule::Machine::iterator j = i->begin(); j != i->end(); ++j )
			j->task = tasks[order[j->task].vertex].second;
	return out;
}

template< class DefaultStructs > template< typename GraphType, typename TaskIterator >
	int SchedulingPar< DefaultStructs >::precLiu( TaskIterator begin, TaskIterator end, const GraphType& DAG,
		Schedule &schedule )
{
	koalaAssert( schedule.getMachNo() == 1,AlgExcWrongArg );
	typedef std::pair< typename GraphType::PVertex,int > Pair;


	int n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator) n++;
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,Element< Task< GraphType > > >::Type
		tasks( n );

	n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ )
	{
		Element< Task< GraphType > > &element = tasks[iterator->vertex] = Element< Task< GraphType > >( *iterator,n );
		element.duedate = iterator->duedate, element.degree = DAG.getEdgeNo( iterator->vertex,EdDirIn );
	}

	typename GraphType::PVertex LOCALARRAY( vertices,n );
	Koala::DAGAlgsPar<DefaultStructs>::topOrd( DAG,vertices );

	SimplArrPool<typename DefaultStructs::template
        HeapCont< Pair,compareSecondFirst< Pair > >::NodeType > alloc(n+1);
    typename DefaultStructs::template
        HeapCont< Pair,compareSecondFirst< Pair > >::Type candidate(&alloc), active(&alloc);

	for( int i = n - 1; i >= 0; i-- )
	{
		typename GraphType::PVertex v = vertices[i];
		Element< Task< GraphType > > &first = tasks[v];

		for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirIn ); e; e = DAG.getEdgeNext( v,e,EdDirIn ) )
		{
			Element< Task< GraphType > > &second = tasks[DAG.getEdgeEnd( e,v )];
			if (second.duedate > first.duedate) second.duedate = first.duedate;
		}

		if (first.degree == 0) candidate.push( Pair( v,first.task.release ) );
	}

	int time = 0, out = std::numeric_limits< int >::min();
	while (!candidate.empty())
	{
		typename GraphType::PVertex v = candidate.top().first;
		Element< Task< GraphType > > &activated = tasks[v];
		active.push( Pair( v,activated.duedate ) );
		candidate.pop();

		if (time < activated.task.release) time = activated.task.release;

		int stop = candidate.empty() ? std::numeric_limits< int >::max() : candidate.top().second;

		while (!active.empty() && time < stop)
		{
			typename GraphType::PVertex v = active.top().first;
			Element< Task< GraphType > > &element = tasks[v];

			int length = std::min( element.timeleft,stop - time );
			time += length, element.timeleft -= length;

			if (schedule.machines[0].empty() || schedule.machines[0].back().task != element.index)
				schedule.machines[0].push_back( TaskPart( element.index,time - length,time,element.parts++ ) );
			else schedule.machines[0].back().end += length;

			if (element.timeleft == 0)
			{
				int delay = time - element.task.duedate;
				if (out < delay) out = delay;

				typename GraphType::PVertex u;
				for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirOut ); e; e = DAG.getEdgeNext( v,e,EdDirOut ) )
				{
					Element< Task< GraphType > > &next = tasks[u = DAG.getEdgeEnd( e,v )];
					if (!(--next.degree)) candidate.push( Pair( u,next.task.release ) );
				}
				active.pop();
			}
		}
	}

	return out;
}

template< class DefaultStructs > template< typename GraphType, typename TaskIterator >
	int SchedulingPar< DefaultStructs >::brucker( TaskIterator begin, TaskIterator end, const GraphType& DAG,
		Schedule &schedule )
{
	typedef std::pair< typename GraphType::PVertex,int > Pair;

	int n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator) n++ ;
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,Element< Task< GraphType > > >::Type
		tasks( n );

	n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ )
	{
		Element< Task< GraphType> > &element = tasks[iterator->vertex] = Element< Task< GraphType > >( *iterator,n );
		element.priority = 1 - iterator->duedate, element.degree = DAG.getEdgeNo( iterator->vertex,EdDirIn );
	}

	// Computes order of vertices
	typename GraphType::PVertex LOCALARRAY( vertices,n );
	Koala::DAGAlgsPar<DefaultStructs>::topOrd( DAG,vertices );

	// Update of deadlines according to prec
	SimplArrPool<typename DefaultStructs::template
        HeapCont< Pair,compareSecondLast< Pair > >::NodeType > alloc(n);
    typename DefaultStructs::template
        HeapCont< Pair,compareSecondLast< Pair > >::Type active(&alloc);

	for( int i = n - 1; i >= 0; i-- )
	{
		typename GraphType::PVertex v = vertices[i];
		Element< Task< GraphType > > &first = tasks[v];
		int priority = first.priority + 1;

		for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirIn ); e; e = DAG.getEdgeNext( v,e,EdDirIn) )
		{
			Element< Task< GraphType > > &second = tasks[DAG.getEdgeEnd( e,v )];
			if (second.priority < priority) second.priority = priority;
		}

		if (first.degree == 0) active.push( Pair( v,first.priority ) );
	}

	int used, time = 0, out = std::numeric_limits< int >::min();
	typename GraphType::PVertex LOCALARRAY( current,schedule.getMachNo() );
	while (!active.empty())
	{
		time++;

		// Assignment of tasks to machines
		for( used = 0; used < schedule.getMachNo() && !active.empty(); used++ )
		{
			Element< Task< GraphType > > &element = tasks[current[used] = active.top().first];
			schedule.machines[used].push_back( TaskPart( element.index,time - 1, time ) );
			out = std::max( time - element.task.duedate,out ), active.pop();
		}

		// Adding of new task to queue
		for( int i = 0; i < used; i++ )
		{
			typename GraphType::PVertex u, v = current[i];
			for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirOut ); e; e = DAG.getEdgeNext( v,e,EdDirOut ) )
			{
				Element< Task< GraphType > > &element = tasks[u = DAG.getEnd( e,v )];
				if (--element.degree == 0) active.push( Pair( u,element.priority ) );
			}
		}
	}

	return out;
}

template< class DefaultStructs > template< typename GraphType, typename TaskIterator >
	int SchedulingPar< DefaultStructs >::hu( TaskIterator begin, TaskIterator end, const GraphType& DAG,
		Schedule &schedule )
{
	typedef std::pair< typename GraphType::PVertex,int > Pair;

	int n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator) n++ ;
	typename DefaultStructs::template AssocCont< typename GraphType::PVertex,Element< Task< GraphType > > >::Type
		tasks( n );

	n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ )
	{
		Element< Task< GraphType > > &element = tasks[iterator->vertex] = Element< Task< GraphType > >( *iterator,n );
		element.priority = 0, element.degree = DAG.getEdgeNo( iterator->vertex,EdDirIn );
	}

	// computes order of vertices
	typename GraphType::PVertex LOCALARRAY( vertices,n );
	Koala::DAGAlgsPar<DefaultStructs>::topOrd( DAG,vertices );

	// Priority queue
	SimplArrPool<typename DefaultStructs::template
        HeapCont<  Pair,compareSecondLast< Pair > >::NodeType> alloc(n);
	typename DefaultStructs::template
        HeapCont<  Pair,compareSecondLast< Pair > >::Type active(&alloc);

	for( int i = n - 1; i >= 0; i-- )
	{
		typename GraphType::PVertex v = vertices[i];
		Element< Task< GraphType > > &first = tasks[v];
		int priority = first.priority + 1;

		for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirIn ); e; e = DAG.getEdgeNext( v,e,EdDirIn ) )
		{
			Element< Task< GraphType > > &second = tasks[DAG.getEdgeEnd( e,v )];
			if (second.priority < priority) second.priority = priority;
		}

		if (first.degree == 0) active.push( Pair( v,first.priority ) );
	}

	int used, time = 0;
	typename GraphType::PVertex LOCALARRAY( current,schedule.getMachNo() );
	while (!active.empty())
	{
		time++;

		// Assignment of tasks to machines
		for( used = 0; used < schedule.getMachNo() && !active.empty(); used++ )
		{
			schedule.machines[used].push_back( TaskPart( tasks[current[used] = active.top().first].index,time - 1,time ) );
			active.pop();
		}

		// Adding of new tasks to queue
		for( int i = 0; i < used; i++ )
		{
			typename GraphType::PVertex u, v = current[i];
			for( typename GraphType::PEdge e = DAG.getEdge( v,EdDirOut ); e; e = DAG.getEdgeNext( v,e,EdDirOut ) )
			{
				Element< Task< GraphType > > &element = tasks[u = DAG.getEnd( e,v )];
				if (--element.degree == 0) active.push( Pair( u,element.priority ) );
			}
		}
	}

	return time;
}

template< class DefaultStructs > template< typename TaskIterator >
	int SchedulingPar< DefaultStructs >::spt( TaskIterator begin, TaskIterator end, Schedule &schedule )
{
	int n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ ) ;

	int LOCALARRAY( tasks,n );
	Scheduling::sortSPT( begin,end,tasks );

	int LOCALARRAY( length,n );
	n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ ) length[n] = iterator->length;

	typename Schedule::Type::iterator iter = schedule.machines.begin() + (n % schedule.getMachNo() ?
		schedule.getMachNo() - n % schedule.getMachNo() : 0);
	int out = 0;
	for( int i = 0; i < n; i++ )
	{
		int time = iter->empty() ? 0 : iter->back().end;
		iter->push_back( TaskPart( tasks[i],time,time + length[tasks[i]] ) ), out += time + length[tasks[i]];
		if (++iter == schedule.machines.end()) iter = schedule.machines.begin();
	}
	return out;
}

template< class DefaultStructs > template< typename TaskIterator >
	int SchedulingPar< DefaultStructs >::hodgson( TaskIterator begin, TaskIterator end, Schedule &schedule )
{
	koalaAssert( schedule.getMachNo() == 1,AlgExcWrongArg );
	typedef std::pair< int,int > Pair;

	int n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ ) ;

	int LOCALARRAY( tasks,n );
	Scheduling::sortEDD( begin,end,tasks );

	HodgsonElement LOCALARRAY( info,n );
	n = 0;
	for( TaskIterator iterator = begin; iterator != end; ++iterator,n++ )
		info[n] = HodgsonElement( n,iterator->length,iterator->duedate );

	SimplArrPool<typename DefaultStructs::template
        HeapCont< Pair,compareSecondLast< Pair > >::NodeType > alloc(n);
    typename DefaultStructs::template
        HeapCont< Pair,compareSecondLast< Pair > >::Type active(&alloc);

	int out = 0, sum = 0;
	for( int i = 0; i < n; i++ )
	{
		HodgsonElement &element = info[tasks[i]];
		sum += element.length;
		active.push( Pair( element.index,element.length ) );
		if (sum > element.duedate)
		{
			sum -= active.top().second, info[active.top().first].late = 1, out++;
			active.pop();
		}
	}

	int time = 0;
	for( int i = 0; i < n; i++ )
	{
		HodgsonElement &element = info[tasks[i]];
		if (!element.late)
			schedule.machines[0].push_back( TaskPart( element.index,time,time + element.length ) ), time += element.length;
	}
	for( int i = 0; i < n; i++ )
		if (info[i].late)
			schedule.machines[0].push_back( TaskPart( i,time,time + info[i].length) ), time += info[i].length;

	return out;
}
