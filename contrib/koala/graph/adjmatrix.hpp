namespace Privates {

// AdjMatrix

template< class VertInfo, class EdgeInfo, class Settings >
	void AdjMatrix< VertInfo,EdgeInfo,Settings,true >::clear()
{
	dirs.clear();
	undirs.clear();
}

//template< class VertInfo, class EdgeInfo, class Settings >
//	void AdjMatrix< VertInfo,EdgeInfo,Settings,true >::defrag()
//{
//	dirs.defrag();
//	undirs.defrag();
//}

template< class VertInfo, class EdgeInfo, class Settings >
	void AdjMatrix< VertInfo,EdgeInfo,Settings,true >::reserve( int size )
{
	dirs.reserve( size );
	undirs.reserve( size );
}

template< class VertInfo, class EdgeInfo, class Settings >
	void AdjMatrix< VertInfo,EdgeInfo,Settings,true >::delVert( Koala::Vertex< VertInfo,EdgeInfo,Settings > *u )
{
	if (Settings::EdAllow & EdUndir) undirs.delInd( u );
	if (Settings::EdAllow & (EdDirIn|EdDirOut)) dirs.delInd( u );
}

template< class VertInfo, class EdgeInfo, class Settings > void
	AdjMatrix< VertInfo,EdgeInfo,Settings,true >::add( Edge< VertInfo,EdgeInfo,Settings > *edge )
{
	if (!edge) return;
	if (edge->type == Directed)
	{
		std::pair< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex,
			typename Graph< VertInfo,EdgeInfo,Settings >::PVertex > ends( edge->vert[0].vert,edge->vert[1].vert );
		Privates::AdjMatrixParals< VertInfo,EdgeInfo,Settings > &pole = dirs( ends.first,ends.second );

		edge->pParal() = pole.last;
		edge->nParal() = NULL;
		if (edge->pParal()) ((typename Graph< VertInfo,EdgeInfo,Settings >::PEdge)edge->pParal())->nParal() = edge;
		else pole.first = edge;
		pole.last = edge;
		pole.degree++;
	}
	else if (edge->type == Undirected)
	{
		std::pair< typename Graph< VertInfo,EdgeInfo,Settings >::PVertex,
			typename Graph< VertInfo,EdgeInfo,Settings >::PVertex > ends( edge->vert[0].vert,edge->vert[1].vert );
		Privates::AdjMatrixParals< VertInfo,EdgeInfo,Settings > &pole = undirs( ends.first,ends.second );
		edge->pParal() = pole.last;
		edge->nParal() = NULL;
		if (edge->pParal()) ((typename Graph< VertInfo,EdgeInfo,Settings >::PEdge)edge->pParal())->nParal() = edge;
		else pole.first = edge;
		pole.last = edge;
		pole.degree++;
	}
}

}
